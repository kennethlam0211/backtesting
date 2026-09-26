"""PPO v2: Vectorized two-actor PPO, 60min decisions, 1min internal stepping.

Based on deprecated/training/ppo.py vectorized design.
Adapted for v2 architecture:
    - env.step() runs ~60 1min bars internally, returns at 60min boundary
    - Two actors: model selector (γ=0.99) + K predictor (γ=0.97)
    - Joint action: [k_index, conf_s4, conf_cgf, conf_rofs] via Beta distributions
    - SyncVectorEnv for parallel data collection
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import gym
from gym.vector import SyncVectorEnv
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import TradingNetwork
import params
from utils.app_logger import get_logger
logger = get_logger('ppo', level=params.LOG_LEVEL)


# ==================== Reward Normalization ====================

class RunningMeanStd:
    """Welford's online algorithm for tracking running mean/variance."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        return max(np.sqrt(self.var), 0.1)


# ==================== PPO Buffer ====================

class PPOBuffer:
    """Vectorized buffer for 60min decision points, two value heads."""
    def __init__(self, num_envs, max_steps, obs_shapes,
                 gamma_selector=0.99, gamma_k=0.97, lam=0.95):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.gamma_selector = gamma_selector
        self.gamma_k = gamma_k
        self.lam = lam
        self.obs_shapes = obs_shapes
        self.ptr = 0

        # Observations: (max_steps, num_envs, *shape)
        self.obs = {key: np.zeros((max_steps, num_envs, *shape), dtype=np.float32)
                    for key, shape in obs_shapes.items()}

        # Actions + values: (max_steps, num_envs)
        self.k_index = np.zeros((max_steps, num_envs), dtype=np.float32)
        # Per-family confidence samples — dict keyed by params.STRAT_FAMILIES
        self.conf_by_family = {fam: np.zeros((max_steps, num_envs), dtype=np.float32)
                               for fam in params.STRAT_FAMILIES}
        self.k_log_probs = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.conf_log_probs = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.pm_rewards = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.selector_rewards = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.conf_rewards = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.k_rewards = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.selector_values = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.conf_values = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.k_values = np.zeros((max_steps, num_envs), dtype=np.float32)

        # Computed — 3 GAEs
        self.selector_advantages = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.selector_returns = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.conf_advantages = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.conf_returns = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.k_advantages = np.zeros((max_steps, num_envs), dtype=np.float32)
        self.k_returns = np.zeros((max_steps, num_envs), dtype=np.float32)

        # Display raw reward sums (for interpretable log output)
        self._pm_display_sum = 0.0
        self._sel_display_sum = 0.0
        self._conf_display_sum = 0.0
        self._k_display_sum = 0.0
        self._display_cnt = 0
        self._final_size_sum = 0.0
        self._final_size_cnt = 0
        self._ep_count = 0
        self._done_count = 0
        self._step_count = 0
        self._final_snapshots = []   # captured before each auto-reset (Gymnasium final_info)
        self._final_size_sq_sum = 0.0   # for std_size in display

    def store(self, obs_dict, k_index, conf_by_family, k_log_prob, conf_log_prob,
             selector_value, conf_value, k_value, pm_reward, sel_reward, conf_reward, k_reward, done):
        """conf_by_family: dict {fam: tensor/ndarray of shape (num_envs,)} — one entry per params.STRAT_FAMILIES."""
        t = self.ptr
        for key in self.obs:
            self.obs[key][t] = obs_dict[key]
        self.k_index[t] = k_index
        for fam, vals in conf_by_family.items():
            self.conf_by_family[fam][t] = vals
        self.k_log_probs[t] = k_log_prob
        self.conf_log_probs[t] = conf_log_prob
        self.pm_rewards[t] = pm_reward
        self.selector_rewards[t] = sel_reward
        self.conf_rewards[t] = conf_reward
        self.k_rewards[t] = k_reward
        self.dones[t] = done
        self.selector_values[t] = selector_value
        self.conf_values[t] = conf_value
        self.k_values[t] = k_value
        self.ptr += 1

    def _gae(self, rewards, values, last_values, gamma):
        """Compute GAE for one value head."""
        n = self.ptr
        advantages = np.zeros((n, self.num_envs), dtype=np.float32)
        for env_i in range(self.num_envs):
            vals = np.append(values[:n, env_i], last_values[env_i])
            last_gae = 0.0
            for t in reversed(range(n)):
                delta = rewards[t, env_i] + gamma * vals[t+1] * (1 - self.dones[t, env_i]) - vals[t]
                last_gae = delta + gamma * self.lam * (1 - self.dones[t, env_i]) * last_gae
                advantages[t, env_i] = last_gae
        return advantages

    def compute_gae(self, last_sel_values, last_conf_values, last_k_values):
        """Compute GAE for 3 value heads."""
        n = self.ptr
        self.selector_advantages[:n] = self._gae(self.selector_rewards, self.selector_values, last_sel_values, self.gamma_selector)
        self.selector_returns[:n] = self.selector_advantages[:n] + self.selector_values[:n]

        self.conf_advantages[:n] = self._gae(self.conf_rewards, self.conf_values, last_conf_values, self.gamma_selector)
        self.conf_returns[:n] = self.conf_advantages[:n] + self.conf_values[:n]

        self.k_advantages[:n] = self._gae(self.k_rewards, self.k_values, last_k_values, self.gamma_k)
        self.k_returns[:n] = self.k_advantages[:n] + self.k_values[:n]

    def get_flat_data(self):
        """Flatten (steps, envs) → (steps*envs) for training."""
        n = self.ptr
        flat = {}
        for key in self.obs:
            flat[f'obs_{key}'] = self.obs[key][:n].reshape(-1, *self.obs[key].shape[2:])
        flat['k_index'] = self.k_index[:n].reshape(-1)
        # Per-family confidence samples flattened to (N,) each
        for fam, arr in self.conf_by_family.items():
            flat[f'conf_{fam.lower()}'] = arr[:n].reshape(-1)
        flat['old_k_log_probs'] = self.k_log_probs[:n].reshape(-1)
        flat['old_conf_log_probs'] = self.conf_log_probs[:n].reshape(-1)
        flat['selector_advantages'] = self.selector_advantages[:n].reshape(-1)
        flat['selector_returns'] = self.selector_returns[:n].reshape(-1)
        flat['conf_advantages'] = self.conf_advantages[:n].reshape(-1)
        flat['conf_returns'] = self.conf_returns[:n].reshape(-1)
        flat['k_advantages'] = self.k_advantages[:n].reshape(-1)
        flat['k_returns'] = self.k_returns[:n].reshape(-1)
        flat['old_sel_values'] = self.selector_values[:n].reshape(-1)
        flat['old_conf_values'] = self.conf_values[:n].reshape(-1)
        flat['old_k_values'] = self.k_values[:n].reshape(-1)
        return flat


# ==================== PPO Trainer ====================

class PPOTrainer:
    """Vectorized two-actor PPO for v2 architecture."""

    def __init__(self, make_train_env, make_val_env=None, make_test_env=None,
                 network=None, device='auto',
                 num_train_envs=8, num_val_envs=2, num_test_envs=1,
                 steps_per_env=64, total_updates=1000,
                 gamma_selector=0.99, gamma_k=0.97, gae_lambda=0.95,
                 lr=3e-4, weight_decay=1e-4,
                 clip_epsilon=0.1, value_coef=0.5, entropy_coef=0.01,
                 l1_coef=1e-5, max_grad_norm=0.5,
                 n_epochs=10, batch_size=256, target_kl=0.02,
                 validation_freq=5, checkpoint_freq=50, early_stop_patience=999,
                 val_steps_per_env=112,
                 total_train_steps=0):

        self.num_train_envs = num_train_envs
        self.num_val_envs = num_val_envs
        self.steps_per_env = steps_per_env
        self.total_train_steps = total_train_steps  # total 60min steps in all train chunks
        self.val_steps_per_env = val_steps_per_env
        self.total_updates = total_updates
        self.gamma_selector = gamma_selector
        self.gamma_k = gamma_k
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.l1_coef = l1_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.initial_kl = 0.5
        self.current_kl = self.initial_kl
        self.validation_freq = validation_freq
        self.checkpoint_freq = checkpoint_freq
        self.early_stop_patience = early_stop_patience
        self.base_lr = lr
        self.warmup_updates = 5

        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Vectorized envs
        self.train_envs = SyncVectorEnv([lambda i=i: make_train_env(i) for i in range(num_train_envs)])
        self.val_envs = SyncVectorEnv([lambda i=i: make_val_env(i) for i in range(num_val_envs)]) if make_val_env else None
        self.test_envs = SyncVectorEnv([lambda i=i: make_test_env(i) for i in range(num_test_envs)]) if make_test_env else None

        # Get obs shapes from single env
        single_env = make_train_env(0)
        self.obs_shapes = single_env.feature_shape
        single_env.close()

        # Network
        self.network = (network or TradingNetwork(selector_mode=params.SELECTOR_MODE)).to(self.device)
        # Separate weight decay: exclude biases, LayerNorm, PReLU slopes, InputGate
        decay_params = []
        no_decay_params = []
        for name, param in self.network.named_parameters():
            if 'bias' in name or '.ln.' in name or 'layernorm' in name.lower() or \
               '.gate' in name or name.endswith('.weight') and param.dim() == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        self.optimizer = optim.Adam([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0},
        ], lr=lr, eps=1e-5)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6)

        # Buffers
        self.train_buffer = PPOBuffer(
            num_train_envs, steps_per_env, self.obs_shapes,
            gamma_selector=gamma_selector, gamma_k=gamma_k, lam=gae_lambda)

        # Per-actor reward normalisation
        self.pm_rms = RunningMeanStd()
        self.sel_rms = RunningMeanStd()
        self.conf_rms = RunningMeanStd()
        self.k_rms = RunningMeanStd()

        # Stateful rollouts
        self._train_obs = None

        # Logging
        self._best_val_score = -float('inf')
        self._no_improve = 0
        self._val_obs = None  # stateful val: keep env state between val calls
        self._val_steps_done = 0  # cumulative val steps for val cyc
        self.training_dir = os.path.dirname(os.path.abspath(__file__))
        self.run_name = f'ppo_{params.TRAINING_MODE}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.model_dir = os.path.join(self.training_dir, 'models', self.run_name)
        self.runs_dir = os.path.join(self.training_dir, 'runs', self.run_name)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.runs_dir)

        # Stats
        self.training_stats = {
            'best_val_loss': float('inf'),
            'best_metric': -float('inf'),
            'no_improvement_count': 0,
        }

        logger.info(f'{"="*70}')
        logger.info(f'PPO v2 Vectorized — {self.run_name}')
        logger.info(f'{"="*70}')
        logger.info(f'Device: {self.device}')
        logger.info(f'Envs: {num_train_envs} train, {num_val_envs} val')
        logger.info(f'Network: {self.network.get_param_count():,} params')
        logger.info(f'Steps/env: {steps_per_env} | Updates: {total_updates}')
        logger.info(f'Gamma: sel={gamma_selector} k={gamma_k} | Clip={clip_epsilon}')
        logger.info(f'LR: {lr} | Batch: {batch_size} | Epochs: {n_epochs}')
        logger.info('Legend: ADTP=AvgDaysToPass  ADTF=AvgDaysToFail  '
                    'ADTPay=AvgDaysToPayout  ADTFAct=AvgDaysToFailActivation  '
                    'DTHR=DailyTargetHitRate  ERPD=ExamResetsPerDay  '
                    'Conf shown as mean(std) per family')
        logger.info(f'{"="*70}')

    def _obs_to_tensor(self, obs_dict):
        """Convert vectorized obs dict to tensors."""
        return {key: torch.as_tensor(arr, dtype=torch.float32, device=self.device)
                for key, arr in obs_dict.items()}

    def _split_obs(self, obs_t):
        """Split obs into selector and K predictor obs. Iterates params.STRAT_FAMILIES.
        regime_features is shared — fed to BOTH heads (each encodes via shared regime_encoder)."""
        sel_keys = ['session_summary', 'pfolio_info', 'regime_features']
        sel_keys += [f'{fam}_variants' for fam in params.STRAT_FAMILIES]
        if params.SELECTOR_MODE == 'rl':
            sel_keys += [f'{fam}_metrics_ts' for fam in params.STRAT_FAMILIES]
        sel = {k: obs_t[k] for k in sel_keys if k in obs_t}
        k = {k: obs_t[k] for k in ['vol_60', 'regime_features'] if k in obs_t}
        return sel, k

    def _format_log(self, update, s, metrics, avg_pm_r, avg_sel_r, avg_conf_r, avg_k_r, mode, lr, elapsed, is_val=False):
        prefix = '    VAL  |' if is_val else f'[{update+1:3d}/{self.total_updates}]'
        progress = (f' cyc:{s.get("cyc", 0):.2f} done_cnt:{s.get("done_cnt", 0):>3} '
                    f'ttl_day:{s.get("ttl_days", 0):>4}')

        if mode == 'train_exam':
            mode_stats = (
                f'PR:{s.get("exam_PR", 0):.2f} Ex:{s.get("exam_success", 0):>3.0f}/{s.get("exam_counts", 0):<3.0f} '
                f'ADTP:{s.get("avg_days_to_pass", 0):.1f} ADTF:{s.get("avg_days_to_fail", 0):.1f} '
                f'DTHR:{s.get("exam_daily_target_hit_rate", 0):.2f} ERPD:{s.get("exam_resets_per_day", 0):.1f}')
            pnl_stats = f'Acc:{s.get("acc_pnl", 0):>+7,.0f} Cost:{s.get("cost", 0):>7,.0f} Rsts:{s.get("total_resets", 0):>2.0f}'
            trade_stats = f'Tr:{s.get("trade_delta", 0):>4.0f} ({s.get("trade_cnt", 0):>4.0f})'
        elif mode == 'train_activation':
            mode_stats = (
                f'ActPR:{s.get("activation_PR", 0):.2f} PayCnt:{s.get("payout_cnt", 0):>3.0f}/{s.get("activation_cnt", 0):<3.0f} '
                f'ADTPay:{s.get("avg_days_to_payout", 0):>5.1f} ADTFAct:{s.get("avg_days_to_fail_activation", 0):.1f} '
                f'BtF:{s.get("avg_back_to_fund", 0):.1f}')
            pnl_stats = f'TH:{s.get("take_home", 0):>+8,.0f} Pay:{s.get("payouts", 0):>+8,.0f} Cost:{s.get("cost", 0):>7,.0f} Acc:{s.get("acc_pnl", 0):>+7,.0f} LWM:{s.get("lwm", 0):>+7,.0f}'
            trade_stats = f'Tr:{s.get("trade_delta", 0):>4.0f} ({s.get("trade_cnt", 0):>5.0f})'
        else:  # eval
            mode_stats = (
                f'ExPR:{s.get("exam_PR", 0):.2f} Ex:{s.get("exam_success", 0):>3.0f}/{s.get("exam_counts", 0):<3.0f} '
                f'ADTP:{s.get("avg_days_to_pass", 0):.1f} ADTF:{s.get("avg_days_to_fail", 0):.1f} '
                f'DTHR:{s.get("exam_daily_target_hit_rate", 0):.2f} ERPD:{s.get("exam_resets_per_day", 0):.1f} '
                f'ActPR:{s.get("activation_PR", 0):.2f} PayCnt:{s.get("payout_cnt", 0):>3.0f}/{s.get("activation_cnt", 0):<3.0f} '
                f'ADTPay:{s.get("avg_days_to_payout", 0):>5.1f} ADTFAct:{s.get("avg_days_to_fail_activation", 0):.1f} '
                f'BtF:{s.get("avg_back_to_fund", 0):.1f}')
            pnl_stats = f'TH:{s.get("take_home", 0):>+8,.0f} Pay:{s.get("payouts", 0):>+8,.0f} Cost:{s.get("cost", 0):>7,.0f} Acc:{s.get("acc_pnl", 0):>+7,.0f} LWM:{s.get("lwm", 0):>+7,.0f}'
            trade_stats = f'SL:{s.get("stop_cnt", 0):>4.0f} Tr:{s.get("trade_cnt", 0):>4.0f} Cm:{s.get("CM", 0):>+7,.0f}'

        loss_stats = f'L:{metrics["loss"]:>8.1f}           ' if is_val else f'L:{metrics["loss"]:>8.1f} KL:{metrics["kl"]:+.4f}'
        rewards = f'Rp:{avg_pm_r:>+6.3f} rs:{avg_sel_r:>+6.3f} rc:{avg_conf_r:>+6.3f} rk:{avg_k_r:>+6.3f}'
        # Short 2-char family tag → avg_conf(std). Iterate STRAT_FAMILIES so log auto-extends.
        fam_conf = ' '.join(
            f'{fam[:2]}:{s.get(f"avg_conf_{fam.lower()}", 0):.2f}({s.get(f"std_conf_{fam.lower()}", 0):.2f})'
            for fam in params.STRAT_FAMILIES)
        actions = (f'K:{s.get("avg_k", 0):>3.0f}({s.get("std_k", 0):>3.0f}) '
                   f'Sz:{s.get("avg_size", 0):.1f}({s.get("std_size", 0):.1f}) ' + fam_conf)
        tail = f' | {lr:.0e} {elapsed:.0f}s' if not is_val else ''

        return f'{prefix}{progress} | {mode_stats} | {pnl_stats} | {trade_stats} | {loss_stats} | {rewards} | {actions}{tail}'

    def collect_trajectories(self, envs, buffer, is_training=True):
        """Collect fixed-length rollouts from vectorized envs."""
        num_envs = envs.num_envs
        buffer.ptr = 0

        if is_training:
            if self._train_obs is None:
                current_obs, _ = envs.reset()
            else:
                current_obs = self._train_obs
        else:
            # Stateful val: keep env state between calls
            if self._val_obs is None:
                current_obs, _ = envs.reset()
            else:
                current_obs = self._val_obs

        self.network.eval()

        n_steps = self.steps_per_env if is_training else self.val_steps_per_env
        for step in range(n_steps):
            obs_t = self._obs_to_tensor(current_obs)
            sel_obs, k_obs = self._split_obs(obs_t)

            with torch.no_grad():
                action_out = self.network.get_action(sel_obs, k_obs, deterministic=not is_training)

            k_idx = action_out['k_index'].cpu().numpy()
            # Per-family confidences and attn — pull from action_out by family name
            conf_by_family_np = {fam: action_out[f'conf_{fam.lower()}'].cpu().numpy()
                                 for fam in params.STRAT_FAMILIES}
            attn_by_family_np = {fam: action_out[f'{fam.lower()}_attn'].cpu().numpy()
                                 for fam in params.STRAT_FAMILIES}
            k_lp = action_out['k_log_prob'].cpu().numpy()
            conf_lp = action_out['conf_log_prob'].cpu().numpy()
            sel_val = action_out['selector_value'].squeeze(-1).cpu().numpy()
            conf_val = action_out['confidence_value'].squeeze(-1).cpu().numpy()
            k_val = action_out['k_value'].squeeze(-1).cpu().numpy()

            # Build action array (num_envs, 1 + n_families)
            conf_stack = [conf_by_family_np[fam] for fam in params.STRAT_FAMILIES]
            actions = np.stack([k_idx] + conf_stack, axis=-1).astype(np.float32)

            # Pass attention weights to envs (selector's actual selection)
            for env_i, env in enumerate(envs.envs):
                for fam in params.STRAT_FAMILIES:
                    env.family_attns[fam] = attn_by_family_np[fam][env_i]

            # Step all envs (each runs ~60 1min bars internally)
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated).astype(np.float32)
            if not hasattr(buffer, '_ep_count'):
                buffer._ep_count = 0
                buffer._done_count = 0
                buffer._step_count = 0
                buffer._final_snapshots = []   # list of dicts captured before each auto-reset
            buffer._ep_count += int(dones.sum())
            buffer._done_count += int(dones.sum())
            buffer._step_count += num_envs  # actual steps collected

            # Capture final_episode_stats snapshots from auto-reset envs (Gymnasium API).
            # When an env hits done, SyncVectorEnv auto-resets it and exposes the
            # pre-reset info via infos['final_info'] (with mask infos['_final_info']).
            fin_mask = infos.get('_final_info')
            fin_arr  = infos.get('final_info')
            if fin_mask is not None and fin_arr is not None:
                for ei, m in enumerate(fin_mask):
                    if m and fin_arr[ei] is not None:
                        snap = fin_arr[ei].get('final_episode_stats')
                        if snap is not None:
                            buffer._final_snapshots.append(snap)

            # Extract per-actor rewards from infos (SyncVectorEnv returns dict of arrays)
            pm_raw = np.array(infos.get('r_pm', np.zeros(num_envs)), dtype=np.float32)
            sel_raw = np.array(infos.get('r_selector_display', np.zeros(num_envs)), dtype=np.float32)
            conf_raw = np.array(infos.get('r_confidence', np.zeros(num_envs)), dtype=np.float32)
            k_raw = np.array(infos.get('r_k', np.zeros(num_envs)), dtype=np.float32)

            # Track RAW display sums for all 4 rewards (interpretable magnitudes)
            if not hasattr(buffer, '_pm_display_sum'):
                buffer._pm_display_sum = 0.0
                buffer._sel_display_sum = 0.0
                buffer._conf_display_sum = 0.0
                buffer._k_display_sum = 0.0
                buffer._display_cnt = 0
            buffer._pm_display_sum += pm_raw.sum()
            buffer._sel_display_sum += sel_raw.sum()
            buffer._conf_display_sum += conf_raw.sum()
            buffer._k_display_sum += k_raw.sum()
            buffer._display_cnt += len(pm_raw)

            # Track average final_size (non-zero only) across steps and envs
            final_sizes = np.array(infos.get('final_size', np.zeros(num_envs)), dtype=np.float32)
            nz_mask = final_sizes != 0
            if not hasattr(buffer, '_final_size_sum'):
                buffer._final_size_sum = 0.0
                buffer._final_size_cnt = 0
            buffer._final_size_sum += final_sizes[nz_mask].sum()
            buffer._final_size_sq_sum = getattr(buffer, '_final_size_sq_sum', 0.0) + (final_sizes[nz_mask] ** 2).sum()
            buffer._final_size_cnt += int(nz_mask.sum())

            # Use raw arrays for gradient path (will be normalized below)
            pm_rewards = pm_raw
            sel_rewards = np.array(infos.get('r_selector', np.zeros(num_envs)), dtype=np.float32)
            conf_rewards = conf_raw
            k_rewards = k_raw

            # Normalize each raw reward component to unit variance, then combine
            if is_training:
                self.pm_rms.update(pm_rewards)
                self.sel_rms.update(sel_rewards)
                self.conf_rms.update(conf_rewards)
                self.k_rms.update(k_rewards)
            pm_n = pm_rewards / max(self.pm_rms.var ** 0.5, 1e-8)
            sel_n = sel_rewards / max(self.sel_rms.var ** 0.5, 1e-8)
            conf_n = conf_rewards / max(self.conf_rms.var ** 0.5, 1e-8)
            k_n = k_rewards / max(self.k_rms.var ** 0.5, 1e-8)
            # Recombine: per-actor weighting of PM vs own signal
            pm_rewards = pm_n
            sel_rewards = 0.0 * pm_n + 1.0 * sel_n    # selector: pure selection signal, no PM contamination
            conf_rewards = 0.3 * pm_n + 0.7 * conf_n  # confidence: 30% PM + 70% pure conf
            k_rewards = k_n                            # K: pure, no PM contamination

            buffer.store(current_obs, k_idx, conf_by_family_np, k_lp, conf_lp,
                        sel_val, conf_val, k_val, pm_rewards, sel_rewards, conf_rewards, k_rewards, dones)

            current_obs = next_obs

        # Cache for stateful rollouts
        if is_training:
            self._train_obs = current_obs
        else:
            self._val_obs = current_obs

        # Bootstrap values for GAE
        obs_t = self._obs_to_tensor(current_obs)
        sel_obs, k_obs = self._split_obs(obs_t)
        with torch.no_grad():
            sel_out = self.network.forward_selector(sel_obs)
            k_out = self.network.forward_k(k_obs)
        buffer.compute_gae(
            sel_out['selector_value'].squeeze(-1).cpu().numpy(),
            sel_out['confidence_value'].squeeze(-1).cpu().numpy(),
            k_out['value'].squeeze(-1).cpu().numpy())

        # Collect env stats
        stats = {}
        try:
            envs_list = envs.envs
            mode = envs_list[0].mode

            # Use the latest N snapshots (one per env that hit done & auto-reset);
            # fall back to live env state for envs without a recent snapshot.
            # Each snapshot is the dict returned by env._snapshot_episode_stats().
            n_envs = len(envs_list)
            recent_snaps = list(getattr(buffer, '_final_snapshots', []))[-n_envs:]
            # Pad with live snapshots so we always have one stats source per env.
            while len(recent_snaps) < n_envs:
                recent_snaps.append(envs_list[len(recent_snaps)]._snapshot_episode_stats())
            ss = recent_snaps   # alias: list of dicts, len == n_envs

            # Common stats (all from snapshots — survive auto-reset)
            stats['acc_pnl'] = float(np.mean([s['acc_pnl'] for s in ss]))
            stats['cost'] = float(np.mean([s['topstep_cost'] for s in ss]))
            stats['total_resets'] = float(np.mean([s['total_resets'] for s in ss]))
            if ss:
                stats['_debug_pnl_list_len'] = len(ss[0]['final_pnl_list'])
            stats['stop_cnt'] = float(np.mean([s['stop_loss_count'] for s in ss]))
            stats['trade_cnt'] = float(np.mean([len(s['sim_net_pnl_per_con_list']) for s in ss]))
            stats['trade_delta'] = float(np.mean([
                max(len(s['sim_net_pnl_per_con_list']) - getattr(envs_list[i], '_trade_cnt_prev', 0), 0)
                for i, s in enumerate(ss)
            ]))
            for i, s in enumerate(ss):
                envs_list[i]._trade_cnt_prev = len(s['sim_net_pnl_per_con_list'])
            stats['CM'] = float(np.mean([s['total_commission'] for s in ss]))
            stats['ttl_days'] = int(sum(s.get('ttl_days', 0) for s in ss))   # SUM across envs
            n = buffer.ptr
            k_opts = envs_list[0].K_OPTIONS
            k_values = [k_opts[int(i)] for i in buffer.k_index[:n].flatten()]
            stats['avg_k'] = float(np.mean(k_values))
            stats['std_k'] = float(np.std(k_values))
            # K distribution for display
            from collections import Counter
            k_counts = Counter(k_values)
            stats['k_dist'] = {k: k_counts.get(k, 0) / len(k_values) for k in k_opts}
            # Per-family avg + std confidence (across rollout × envs).
            for fam in params.STRAT_FAMILIES:
                arr = buffer.conf_by_family[fam][:n]
                stats[f'avg_conf_{fam.lower()}'] = float(arr.mean())
                stats[f'std_conf_{fam.lower()}'] = float(arr.std())
            stats['ep_done'] = getattr(buffer, '_ep_count', 0) #acc done count
            stats['done_cnt'] = getattr(buffer, '_done_count', 0) #done count per update
            buffer._done_count = 0
            buffer._final_snapshots = []   # clear after use
            stats['lwm'] = float(np.mean([min(np.cumsum(s['final_pnl_list'])) if s['final_pnl_list'] else 0 for s in ss]))
            stats['cyc'] = getattr(buffer, '_step_count', 0) / max(self.total_train_steps, 1)
            stats['avg_r_pm'] = buffer.pm_rewards[:buffer.ptr].mean() if buffer.ptr > 0 else 0
            cnt = max(buffer._final_size_cnt, 1)
            mean_sz = buffer._final_size_sum / cnt
            stats['avg_size'] = float(mean_sz)
            sq_sum = getattr(buffer, '_final_size_sq_sum', 0.0)
            stats['std_size'] = float(max(sq_sum / cnt - mean_sz ** 2, 0.0) ** 0.5)

            if mode == 'train_exam':
                total_success = sum(s['exam_success'] for s in ss)
                total_exams   = sum(s['exam_counts']  for s in ss)
                # Subtract 1 per env for in-progress (not concluded) exam, unless in activation
                total_in_progress = sum(0 if s['pass_exam'] else 1 for s in ss)
                total_concluded = max(total_exams - total_in_progress, 1)
                stats['exam_PR'] = total_success / total_concluded
                stats['exam_success'] = total_success
                stats['exam_counts'] = total_concluded
                _days = [np.mean(s['days_taken_to_pass_exams']) for s in ss if s['days_taken_to_pass_exams']]
                stats['avg_days_to_pass'] = float(np.mean(_days)) if _days else 0
                _fdays = [np.mean(s['days_taken_to_fail_exams']) for s in ss if s['days_taken_to_fail_exams']]
                stats['avg_days_to_fail'] = float(np.mean(_fdays)) if _fdays else 0
                _hits = [np.mean(s['exam_daily_target_hit_day_list']) for s in ss if s['exam_daily_target_hit_day_list']]
                stats['exam_daily_target_hit_rate'] = float(np.mean(_hits)) if _hits else 0
                _resets = [np.mean(s['exam_daily_resets_list']) for s in ss if s['exam_daily_resets_list']]
                stats['exam_resets_per_day'] = float(np.mean(_resets)) if _resets else 0

            elif mode == 'train_activation':
                total_payouts     = sum(s['payout_cnt']     for s in ss)
                total_activations = sum(s['activation_cnt'] for s in ss)
                total_in_progress = sum(1 if s['pass_exam'] else 0 for s in ss)
                total_concluded = max(total_activations - total_in_progress, 1)
                stats['activation_PR'] = total_payouts / total_concluded
                stats['take_home'] = float(np.mean([s['payouts'] + s['topstep_cost'] for s in ss]))
                stats['payouts']   = float(np.mean([s['payouts']      for s in ss]))
                stats['payout_cnt']     = total_payouts
                stats['activation_cnt'] = total_concluded
                _btf = [np.mean(s['back_to_fund_list']) for s in ss if s['back_to_fund_list']]
                stats['avg_back_to_fund'] = float(np.mean(_btf)) if _btf else 0
                _dpay = [np.mean(s['days_taken_to_payouts']) for s in ss if s['days_taken_to_payouts']]
                stats['avg_days_to_payout'] = float(np.mean(_dpay)) if _dpay else 0
                _fact = [np.mean(s['days_taken_to_fail_activation']) for s in ss if s['days_taken_to_fail_activation']]
                stats['avg_days_to_fail_activation'] = float(np.mean(_fact)) if _fact else 0
            else:  # eval
                stats['payouts']   = float(np.mean([s['payouts']      for s in ss]))
                stats['take_home'] = float(np.mean([s['payouts'] + s['topstep_cost'] for s in ss]))
                total_success = sum(s['exam_success'] for s in ss)
                total_exams   = sum(s['exam_counts']  for s in ss)
                total_in_progress = sum(0 if s['pass_exam'] else 1 for s in ss)
                total_concluded = max(total_exams - total_in_progress, 1)
                stats['exam_PR'] = total_success / total_concluded
                stats['exam_success'] = total_success
                stats['exam_counts'] = total_concluded
                _days = [np.mean(s['days_taken_to_pass_exams']) for s in ss if s['days_taken_to_pass_exams']]
                stats['avg_days_to_pass'] = float(np.mean(_days)) if _days else 0
                _fdays = [np.mean(s['days_taken_to_fail_exams']) for s in ss if s['days_taken_to_fail_exams']]
                stats['avg_days_to_fail'] = float(np.mean(_fdays)) if _fdays else 0
                _fact = [np.mean(s['days_taken_to_fail_activation']) for s in ss if s['days_taken_to_fail_activation']]
                stats['avg_days_to_fail_activation'] = float(np.mean(_fact)) if _fact else 0
                total_payouts     = sum(s['payout_cnt']     for s in ss)
                total_activations = sum(s['activation_cnt'] for s in ss)
                act_in_progress = sum(1 if s['pass_exam'] else 0 for s in ss)
                act_concluded = max(total_activations - act_in_progress, 1)
                stats['activation_PR'] = total_payouts / act_concluded
                stats['payout_cnt']     = total_payouts
                stats['activation_cnt'] = act_concluded
                _btf = [np.mean(s['back_to_fund_list']) for s in ss if s['back_to_fund_list']]
                stats['avg_back_to_fund'] = float(np.mean(_btf)) if _btf else 0
                _dpay = [np.mean(s['days_taken_to_payouts']) for s in ss if s['days_taken_to_payouts']]
                stats['avg_days_to_payout'] = float(np.mean(_dpay)) if _dpay else 0

        except Exception:
            pass

        return stats

    def _curriculum_weights(self, cyc):
        """Curriculum DISABLED — both actors always at full weight from cyc=0.

        Reverted from phased schedule (K-only → ramp → full) because the silenced
        selector/conf actor early on appeared to delay differentiation. Function kept
        in case we want to reintroduce a softer schedule later.
        """
        return 1.0, 1.0
        # ── Previous phased schedule ──
        # if cyc < 0.05:
        #     return 1.0, 0.0
        # if cyc < 0.15:
        #     return 1.0, (cyc - 0.05) / 0.10  # linear ramp
        # return 1.0, 1.0

    def train_epoch(self, data, cyc=1.0):
        """One epoch of PPO updates. cyc is training cycle (0..1) for curriculum."""
        n_samples = len(data['k_index'])
        indices = np.random.permutation(n_samples)
        k_w, conf_w = self._curriculum_weights(cyc)
        # Entropy coefficient: linear decay from `self.entropy_coef` → 0.01 over cyc 0→1,
        # then constant at 0.01. Higher early to explore, lower late to converge.
        ent_floor = 0.01
        ent_coef_now = self.entropy_coef + (ent_floor - self.entropy_coef) * min(cyc, 1.0)

        total_loss = 0
        total_policy = 0
        total_value = 0
        total_entropy = 0
        n_batches = 0
        last_kl = 0
        l1_loss_val = 0

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            idx = indices[start:end]

            # Build obs dicts (regime_features is shared — fed to both heads)
            _sel_keys = ['session_summary', 'pfolio_info', 'regime_features']
            _sel_keys += [f'{fam}_variants' for fam in params.STRAT_FAMILIES]
            if params.SELECTOR_MODE == 'rl':
                _sel_keys += [f'{fam}_metrics_ts' for fam in params.STRAT_FAMILIES]
            sel_obs = {k.replace('obs_', ''): torch.FloatTensor(data[k][idx]).to(self.device)
                      for k in data if k.startswith('obs_') and k.replace('obs_', '') in _sel_keys}
            k_obs = {k.replace('obs_', ''): torch.FloatTensor(data[k][idx]).to(self.device)
                    for k in data if k.startswith('obs_') and k.replace('obs_', '') in
                    ['vol_60', 'regime_features']}

            k_idx = torch.FloatTensor(data['k_index'][idx]).to(self.device)
            conf_by_family = {fam: torch.FloatTensor(data[f'conf_{fam.lower()}'][idx]).to(self.device)
                              for fam in params.STRAT_FAMILIES}
            old_k_lp = torch.FloatTensor(data['old_k_log_probs'][idx]).to(self.device)
            old_conf_lp = torch.FloatTensor(data['old_conf_log_probs'][idx]).to(self.device)

            sel_adv = torch.FloatTensor(data['selector_advantages'][idx]).to(self.device)
            conf_adv = torch.FloatTensor(data['conf_advantages'][idx]).to(self.device)
            k_adv = torch.FloatTensor(data['k_advantages'][idx]).to(self.device)

            # Per-actor advantages:
            #   K actor → pure k_advantage
            #   Confidence actor (ModelSelectorHead) → selector + confidence advantages (owns both)
            conf_combined_adv = (sel_adv + conf_adv) / 2.0
            # Normalize each separately
            k_adv_norm = (k_adv - k_adv.mean()) / (k_adv.std() + 1e-8)
            conf_adv_norm = (conf_combined_adv - conf_combined_adv.mean()) / (conf_combined_adv.std() + 1e-8)

            sel_ret = torch.FloatTensor(data['selector_returns'][idx]).to(self.device)
            conf_ret = torch.FloatTensor(data['conf_returns'][idx]).to(self.device)
            k_ret = torch.FloatTensor(data['k_returns'][idx]).to(self.device)

            # Evaluate
            eval_out = self.network.evaluate_action(sel_obs, k_obs, k_idx, conf_by_family)
            new_k_lp = eval_out['k_log_prob']
            new_conf_lp = eval_out['conf_log_prob']
            k_entropy_t = eval_out['k_entropy']
            conf_entropy_t = eval_out['conf_entropy']
            entropy = k_entropy_t + conf_entropy_t
            sel_val = eval_out['selector_value'].squeeze(-1)
            conf_val = eval_out['confidence_value'].squeeze(-1)
            k_val = eval_out['k_value'].squeeze(-1)

            # Separate PPO clip per actor
            k_ratio = torch.exp(new_k_lp - old_k_lp)
            k_surr1 = k_ratio * k_adv_norm
            k_surr2 = torch.clamp(k_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * k_adv_norm
            k_policy_loss = -torch.min(k_surr1, k_surr2).mean()

            conf_ratio = torch.exp(new_conf_lp - old_conf_lp)
            conf_surr1 = conf_ratio * conf_adv_norm
            conf_surr2 = torch.clamp(conf_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * conf_adv_norm
            conf_policy_loss = -torch.min(conf_surr1, conf_surr2).mean()

            # Curriculum: phase actor weights by cycle (k_w, conf_w from _curriculum_weights)
            policy_loss = k_w * k_policy_loss + conf_w * conf_policy_loss

            # Value losses — 3 heads, clipped (from ppo.py)
            old_sel_val = torch.FloatTensor(data['old_sel_values'][idx]).to(self.device)
            old_conf_val = torch.FloatTensor(data['old_conf_values'][idx]).to(self.device)
            old_k_val = torch.FloatTensor(data['old_k_values'][idx]).to(self.device)

            def clipped_value_loss(val, old_val, ret):
                v_clipped = old_val + torch.clamp(val - old_val, -self.clip_epsilon, self.clip_epsilon)
                return 0.5 * torch.max((val - ret).pow(2), (v_clipped - ret).pow(2)).mean()

            sel_vloss = clipped_value_loss(sel_val, old_sel_val, sel_ret)
            conf_vloss = clipped_value_loss(conf_val, old_conf_val, conf_ret)
            k_vloss = clipped_value_loss(k_val, old_k_val, k_ret)
            # Selector/conf value losses also gated by curriculum so heads don't drift
            # while their policies are silenced.
            value_loss = k_w * k_vloss + conf_w * (sel_vloss + conf_vloss)

            # Entropy + L1 (entropy split: K vs conf, gated by curriculum)
            entropy_loss = -ent_coef_now * (k_w * k_entropy_t.mean() + conf_w * conf_entropy_t.mean())
            l1_loss = self.l1_coef * self.network.l1_first_layer_loss()

            loss = policy_loss + self.value_coef * value_loss + entropy_loss + l1_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            # Skip step if loss is NaN
            if not torch.isnan(loss) and not torch.isinf(loss):
                self.optimizer.step()

            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_entropy += entropy.mean().item()
            l1_loss_val = l1_loss.item()
            n_batches += 1

            with torch.no_grad():
                # KL per actor — break on whichever diverges first
                k_kl = (old_k_lp - new_k_lp).mean().item()
                conf_kl = (old_conf_lp - new_conf_lp).mean().item()
                last_kl = (k_kl + conf_kl) / 2.0
            if abs(k_kl) > 1.5 * self.current_kl or abs(conf_kl) > 1.5 * self.current_kl:
                break

        nb = max(n_batches, 1)
        return {
            'loss': total_loss / nb, 'policy_loss': total_policy / nb,
            'value_loss': total_value / nb, 'entropy': total_entropy / nb,
            'kl': last_kl, 'l1': l1_loss_val,
        }

    def _prewarm_envs(self, envs, label='train'):
        """Fast-forward envs with zero-action steps to fill long-horizon regime buffers.
        No PPO storage; advances env state until env.regime_warmed is True."""
        warmup_steps = max(envs.envs[0]._regime_horizons)
        n = envs.num_envs
        zero_action = np.zeros((n, 1 + len(params.STRAT_FAMILIES)), dtype=np.float32)
        obs, _ = envs.reset()
        logger.info(f'Pre-warming {label} envs ({warmup_steps} hourly steps × {n} envs) to fill regime buffers...')
        t_warm = time.time()
        for _ in range(warmup_steps):
            obs, _, _, _, _ = envs.step(zero_action)
        all_warmed = all(e.regime_warmed for e in envs.envs)
        logger.info(f'Pre-warm done {label}: warmed={all_warmed}  time={time.time()-t_warm:.1f}s')
        return obs

    def train(self):
        """Full training loop."""
        start_time = time.time()

        # Pre-warm regime buffers before any PPO updates.
        self._train_obs = self._prewarm_envs(self.train_envs, label='train')
        if self.val_envs is not None:
            self._val_obs = self._prewarm_envs(self.val_envs, label='val')

        for update in range(self.total_updates):
            t0 = time.time()

            # Cycle progress
            cyc = getattr(self.train_buffer, '_step_count', 0) / max(self.total_train_steps, 1)

            # KL annealing
            progress = min(update / max(self.total_updates * 0.1, 1), 1.0)  # ramp to 1.0 at 10% of training
            self.current_kl = self.initial_kl + (self.target_kl - self.initial_kl) * progress

            # LR: small early, ramp up to full over full cycle
            if update < self.warmup_updates:
                lr_now = self.base_lr * 0.3 * (update + 1) / self.warmup_updates  # warmup to 30% of base
                self.current_kl = 5.0
            elif cyc < 1.0:
                lr_now = self.base_lr * (0.3 + 0.7 * cyc)  # 30% → 100% over full cycle
            else:
                lr_now = self.base_lr  # full LR at end of cycle
            lr_now = max(lr_now, 1e-6)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_now

            # Pass training progress to envs for reward curriculum (full at end of cycle)
            training_progress = min(cyc, 1.0)
            for e in self.train_envs.envs:
                e.training_progress = training_progress
            if self.val_envs:
                for e in self.val_envs.envs:
                    e.training_progress = training_progress

            # Collect
            train_stats = self.collect_trajectories(self.train_envs, self.train_buffer, is_training=True)
            train_data = self.train_buffer.get_flat_data()

            # Train
            self.network.train()
            metrics = {}
            epochs_run = 0
            for epoch in range(self.n_epochs):
                metrics = self.train_epoch(train_data, cyc=cyc)
                epochs_run = epoch + 1
                if abs(metrics['kl']) > 1.5 * self.current_kl:
                    break

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            # ── TensorBoard ──────────────────────────────────────
            self.writer.add_scalar('loss/total', metrics['loss'], update)
            self.writer.add_scalar('loss/policy', metrics['policy_loss'], update)
            self.writer.add_scalar('loss/value', metrics['value_loss'], update)
            self.writer.add_scalar('loss/entropy', metrics['entropy'], update)
            self.writer.add_scalar('loss/kl', metrics['kl'], update)
            self.writer.add_scalar('train/lr', lr, update)
            n = self.train_buffer.ptr
            self.writer.add_scalar('train/r_selector', self.train_buffer.selector_rewards[:n].mean(), update)
            self.writer.add_scalar('train/r_confidence', self.train_buffer.conf_rewards[:n].mean(), update)
            self.writer.add_scalar('train/r_k', self.train_buffer.k_rewards[:n].mean(), update)

            for k, v in train_stats.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    self.writer.add_scalar(f'trade/{k}', v, update)

            # Gate values
            for name, module in self.network.named_modules():
                if hasattr(module, 'gate') and isinstance(module.gate, torch.nn.Parameter):
                    gate = torch.sigmoid(module.gate).detach()
                    self.writer.add_scalar(f'gates/{name}_mean', gate.mean().item(), update)

            # ── Console log (all 4 rewards shown as RAW for interpretability) ──
            n = self.train_buffer.ptr
            d_cnt = max(self.train_buffer._display_cnt, 1)
            avg_pm_r = self.train_buffer._pm_display_sum / d_cnt
            avg_sel_r = self.train_buffer._sel_display_sum / d_cnt
            avg_conf_r = self.train_buffer._conf_display_sum / d_cnt
            avg_k_r = self.train_buffer._k_display_sum / d_cnt
            mode = self.train_envs.envs[0].mode
            s = train_stats

            log_line = self._format_log(update, s, metrics, avg_pm_r, avg_sel_r, avg_conf_r, avg_k_r, mode, lr, elapsed)

            logger.info(log_line)

            # ── Validation ───────────────────────────────────────
            if self.val_envs and (update + 1) % self.validation_freq == 0 and update > 0:
                # Always full-chunk eval from the beginning of val set.
                # Reset val env state every time so each validation pass starts fresh
                # at chunk start — gives stable, comparable PR measurement across updates.
                # Use MIN chunk size across val envs so no env hits done mid-pass and
                # cycles into a fresh chunk with stale (200h) regime buffers from the prior chunk.
                val_chunk_steps = min(
                    (len(e._np_ohlc) // 60) for e in self.val_envs.envs
                    if hasattr(e, '_np_ohlc')
                ) if all(hasattr(e, '_np_ohlc') for e in self.val_envs.envs) else 960
                eval_steps = val_chunk_steps
                self._val_obs = None  # force fresh reset to chunk start
                # ── Previous: stateful warm-up then switch to full-chunk after val_cyc >= 1 ──
                # pre_cyc = self._val_steps_done / max(val_chunk_steps, 1)
                # eval_steps = val_chunk_steps if pre_cyc >= 1.0 else self.val_steps_per_env
                # if pre_cyc >= 1.0:
                #     self._val_obs = None
                val_buffer = PPOBuffer(
                    self.num_val_envs, eval_steps, self.obs_shapes,
                    gamma_selector=self.gamma_selector, gamma_k=self.gamma_k)
                # Temporarily override val_steps_per_env for this collect
                _orig_val_steps = self.val_steps_per_env
                self.val_steps_per_env = eval_steps
                val_stats = self.collect_trajectories(self.val_envs, val_buffer, is_training=False)
                self.val_steps_per_env = _orig_val_steps
                self._val_steps_done += eval_steps
                val_stats['cyc'] = self._val_steps_done / max(val_chunk_steps, 1)

                for k, v in val_stats.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        self.writer.add_scalar(f'val/{k}', v, update)

                val_d_cnt = max(val_buffer._display_cnt, 1)
                val_pm_r = val_buffer._pm_display_sum / val_d_cnt
                val_sel_r = val_buffer._sel_display_sum / val_d_cnt
                val_conf_r = val_buffer._conf_display_sum / val_d_cnt
                val_k_r = val_buffer._k_display_sum / val_d_cnt
                # Val loss (forward only, no backward)
                val_data = val_buffer.get_flat_data()
                val_loss = 0.0
                saved = ''
                if len(val_data['k_index']) > 0:
                    self.network.eval()
                    with torch.no_grad():
                        idx = np.arange(len(val_data['k_index']))
                        _vsel_keys = ['session_summary', 'pfolio_info', 'regime_features']
                        _vsel_keys += [f'{fam}_variants' for fam in params.STRAT_FAMILIES]
                        if params.SELECTOR_MODE == 'rl':
                            _vsel_keys += [f'{fam}_metrics_ts' for fam in params.STRAT_FAMILIES]
                        sel_obs = {k.replace('obs_', ''): torch.FloatTensor(val_data[k][idx]).to(self.device)
                                  for k in val_data if k.startswith('obs_') and k.replace('obs_', '') in _vsel_keys}
                        k_obs = {k.replace('obs_', ''): torch.FloatTensor(val_data[k][idx]).to(self.device)
                                for k in val_data if k.startswith('obs_') and k.replace('obs_', '') in
                                ['vol_60', 'regime_features']}
                        val_conf_by_family = {fam: torch.FloatTensor(val_data[f'conf_{fam.lower()}'][idx]).to(self.device)
                                              for fam in params.STRAT_FAMILIES}
                        eval_out = self.network.evaluate_action(sel_obs, k_obs,
                            torch.FloatTensor(val_data['k_index'][idx]).to(self.device),
                            val_conf_by_family)
                        sel_val = eval_out['selector_value'].squeeze(-1)
                        conf_val = eval_out['confidence_value'].squeeze(-1)
                        k_val = eval_out['k_value'].squeeze(-1)
                        sel_ret = torch.FloatTensor(val_data['selector_returns'][idx]).to(self.device)
                        conf_ret = torch.FloatTensor(val_data['conf_returns'][idx]).to(self.device)
                        k_ret = torch.FloatTensor(val_data['k_returns'][idx]).to(self.device)
                        val_loss = ((sel_val - sel_ret).pow(2).mean() +
                                   (conf_val - conf_ret).pow(2).mean() +
                                   (k_val - k_ret).pow(2).mean()).item()

                    val_pr = val_stats.get('exam_PR', 0)
                    # Combined: 70% PR + 30% loss smoothness
                    val_score = 0.7 * val_pr + 0.3 * (1 / (1 + val_loss))
                    if val_score > self._best_val_score:
                        self._best_val_score = val_score
                        self._no_improve = 0
                        torch.save(self.network.state_dict(), f'{self.model_dir}/best_model.pth')
                        saved = f' *BEST PR:{val_pr:.2f}*'
                    else:
                        self._no_improve += 1
                        saved = ''

                val_metrics = {'loss': val_loss, 'kl': 0}
                val_mode = self.val_envs.envs[0].mode
                val_str = self._format_log(update, val_stats, val_metrics, val_pm_r, val_sel_r, val_conf_r, val_k_r, val_mode, lr, elapsed, is_val=True) + saved
                logger.info(val_str)

                if self._no_improve >= self.early_stop_patience:
                    logger.info(f'Early stop: no val improvement for {self.early_stop_patience} checks')
                    break

            # ── Checkpoint ───────────────────────────────────────
            if self.checkpoint_freq > 0 and (update + 1) % self.checkpoint_freq == 0:
                ckpt = f'{self.model_dir}/checkpoint_{update+1}.pth'
                torch.save(self.network.state_dict(), ckpt)
                logger.info(f'Checkpoint: {ckpt}')

        # Final save
        final_path = f'{self.model_dir}/final_model.pth'
        torch.save(self.network.state_dict(), final_path)

        total_time = time.time() - start_time
        logger.info(f'{"="*70}')
        logger.info(f'Training Complete — {self.run_name}')
        logger.info(f'Time: {total_time:.0f}s | Model: {final_path}')
        logger.info(f'{"="*70}')

        # Save activation params from val stats (for train_activation)
        if self.val_envs and self.train_envs.envs[0].mode == 'train_exam':
            import yaml
            val_envs = self.val_envs.envs
            total_success = sum(e.pm.topstep.exam_success for e in val_envs)
            total_counts = sum(e.pm.topstep.exam_counts for e in val_envs)
            days_pass_lists = [d for e in val_envs for d in e.pm.topstep.days_taken_to_pass_exams]
            activation_params = {
                'exam_success': int(total_success),
                'exam_counts': int(total_counts),
                'pass_rate': round(total_success / max(total_counts, 1), 3),
                'avg_days_to_pass_exam': round(float(np.mean(days_pass_lists)), 1) if days_pass_lists else 10,
                'best_model_path': f'{self.model_dir}/best_model.pth',
            }
            yaml_path = os.path.join(self.model_dir, 'activation_params.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(activation_params, f, default_flow_style=False)
            logger.info(f'Activation params saved: {yaml_path}')
            logger.info(f'  {activation_params}')

        self.writer.close()
        self.train_envs.close()
        if self.val_envs:
            self.val_envs.close()
        if self.test_envs:
            self.test_envs.close()

    def save(self, path, update=None):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update': update,
            'stats': self.training_stats,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and 'network' in ckpt:
            self.network.load_state_dict(ckpt['network'])
            if 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            if 'stats' in ckpt:
                self.training_stats.update(ckpt['stats'])
        else:
            self.network.load_state_dict(ckpt)


if __name__ == '__main__':
    logger.info('PPO v2 Vectorized — use run_ppo.py to train with data.')
