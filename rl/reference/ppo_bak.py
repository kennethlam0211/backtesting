import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_pipeline"))

"""
Vectorized PPO2 (Proximal Policy Optimization) for Discrete Actions
====================================================================

Compatible with TradingNetworkLite (discrete 3-action policy)

Key features:
- Vectorized environments (SyncVectorEnv) for parallel data collection
- Dict observation handling for TradingNetworkLite
- Discrete action space (0=clear, 1=buy, 2=sell) with confidence penalty
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Value function clipping
- Train/Validation split with early stopping
- Learning rate scheduling
"""

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import gym
from gym.vector import SyncVectorEnv
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime
import time
import os
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')





# ==================== Reward Normalization ====================

class RunningMeanStd:
    """Welford's online algorithm for tracking running mean/variance.
    Used to normalize rewards for stable value function learning.
    Prevents value head instability when reward scale varies 100x across episodes
    (e.g. big trending day vs choppy day)."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        # Filter NaN to prevent poisoning running statistics
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        return max(np.sqrt(self.var), 0.1)


# ==================== Multi-Actor PPO2 Buffer with GAE ====================
ACTOR_NAMES = ['direction', 'pm']

class MultiActorPPOBuffer:
    """
    Buffer for 2-actor PPO: direction (3-way), portfolio manager.

    Shared obs buffers + per-actor (act, logp, rew, val, adv, ret) buffers.
    GAE is computed independently per actor using each actor's own rewards/values.
    """
    def __init__(self, num_envs: int, max_len: int, obs_shapes: Dict[str, tuple],
                 gamma: float = 0.99, lam: float = 0.95,
                 actor_gammas: Optional[Dict[str, float]] = None):
        self.num_envs = num_envs
        self.max_len = max_len
        self.gamma = gamma
        self.lam = lam
        # Per-actor gamma: L/S use shorter horizon (signal quality), PM uses longer (session-level)
        self.actor_gammas = actor_gammas or {name: gamma for name in ACTOR_NAMES}
        self.obs_shapes = obs_shapes

        # Shared observation buffers
        self.obs_bufs = {
            key: np.zeros((max_len, num_envs, *shape), dtype=np.float32)
            for key, shape in obs_shapes.items()
        }

        # Per-actor buffers (direction uses int64 for discrete, PM uses float32 for Beta)
        self.actor_bufs = {}
        act_dtypes = {'direction': np.int64, 'pm': np.float32}
        for name in ACTOR_NAMES:
            self.actor_bufs[name] = {
                'act': np.zeros((max_len, num_envs), dtype=act_dtypes[name]),
                'logp': np.zeros((max_len, num_envs), dtype=np.float32),
                'rew': np.zeros((max_len, num_envs), dtype=np.float32),
                'val': np.zeros((max_len, num_envs), dtype=np.float32),
                'adv': np.zeros((max_len, num_envs), dtype=np.float32),
                'ret': np.zeros((max_len, num_envs), dtype=np.float32),
            }

        # PM extra: stop_ratio action and log_prob
        self.actor_bufs['pm']['stop_act'] = np.zeros((max_len, num_envs), dtype=np.float32)
        self.actor_bufs['pm']['stop_logp'] = np.zeros((max_len, num_envs), dtype=np.float32)

        self.done_buf = np.zeros((max_len, num_envs), dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = np.zeros(num_envs, dtype=np.int32)

    def store(self, obs_dict: Dict[str, np.ndarray],
              actor_data: Dict[str, Dict[str, np.ndarray]],
              done: np.ndarray):
        """
        Store one timestep from all environments.

        actor_data: {
            'direction': {'act': ..., 'logp': ..., 'rew': ..., 'val': ...},
            'pm':        {'act': ..., 'stop_act': ..., 'logp': ..., 'stop_logp': ..., 'rew': ..., 'val': ...},
        }
        """
        for key in self.obs_shapes:
            self.obs_bufs[key][self.ptr] = obs_dict[key]

        for name in ACTOR_NAMES:
            for field in ('act', 'logp', 'rew', 'val'):
                self.actor_bufs[name][field][self.ptr] = actor_data[name][field]

        # PM extra fields
        for field in ('stop_act', 'stop_logp'):
            if field in actor_data.get('pm', {}):
                self.actor_bufs['pm'][field][self.ptr] = actor_data['pm'][field]

        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_vals_dict: Dict[str, np.ndarray],
                    env_indices: Optional[np.ndarray] = None):
        """Compute GAE independently for each actor."""
        if env_indices is None:
            env_indices = np.arange(self.num_envs)

        for name in ACTOR_NAMES:
            bufs = self.actor_bufs[name]
            g = self.actor_gammas[name]
            for env_idx in env_indices:
                path_slice = slice(self.path_start_idx[env_idx], self.ptr)

                rews = np.append(bufs['rew'][path_slice, env_idx], last_vals_dict[name][env_idx])
                vals = np.append(bufs['val'][path_slice, env_idx], last_vals_dict[name][env_idx])
                dones = self.done_buf[path_slice, env_idx]

                deltas = rews[:-1] + g * vals[1:] * (1 - dones) - vals[:-1]
                bufs['adv'][path_slice, env_idx] = self._discount_cumsum(deltas, g * self.lam, dones)
                bufs['ret'][path_slice, env_idx] = bufs['adv'][path_slice, env_idx] + vals[:-1]

        for env_idx in env_indices:
            self.path_start_idx[env_idx] = self.ptr

    def get_data_for_envs(self, env_indices: np.ndarray) -> Dict:
        """Get flattened data for specific environments. Returns dict with {actor}_{field} keys."""
        obs_dict = {
            key: buf[:self.ptr, env_indices].reshape(-1, *shape)
            for key, (buf, shape) in zip(
                self.obs_shapes.keys(),
                [(self.obs_bufs[k], self.obs_shapes[k]) for k in self.obs_shapes.keys()]
            )
        }

        result = {'obs': obs_dict}

        for name in ACTOR_NAMES:
            bufs = self.actor_bufs[name]
            result[f'{name}_act'] = bufs['act'][:self.ptr, env_indices].reshape(-1)
            result[f'{name}_logp'] = bufs['logp'][:self.ptr, env_indices].reshape(-1)
            result[f'{name}_ret'] = bufs['ret'][:self.ptr, env_indices].reshape(-1)
            result[f'{name}_val'] = bufs['val'][:self.ptr, env_indices].reshape(-1)

            adv = bufs['adv'][:self.ptr, env_indices].reshape(-1)
            # NOTE: advantages are NOT normalized here — normalization happens
            # per mini-batch in train_epoch to avoid skew from uneven splits
            result[f'{name}_adv'] = adv

        result['pm_stop_act'] = self.actor_bufs['pm']['stop_act'][:self.ptr, env_indices].reshape(-1)
        result['pm_stop_logp'] = self.actor_bufs['pm']['stop_logp'][:self.ptr, env_indices].reshape(-1)
        return result

    def clear_all_data(self, env_indices: np.ndarray):
        """Clear buffer data for specific environment indices."""
        for key in self.obs_bufs:
            self.obs_bufs[key][:, env_indices] = 0
        for name in ACTOR_NAMES:
            for field in self.actor_bufs[name]:
                self.actor_bufs[name][field][:, env_indices] = 0
        self.done_buf[:, env_indices] = 0
        self.path_start_idx[env_indices] = 0

        if len(env_indices) == self.num_envs:
            self.ptr = 0

    def _discount_cumsum(self, x: np.ndarray, discount: float,
                         dones: np.ndarray = None) -> np.ndarray:
        """Compute discounted cumulative sums, resetting at episode boundaries."""
        out = np.zeros_like(x)
        running = 0
        for i in reversed(range(len(x))):
            if dones is not None:
                running = running * (1 - dones[i])
            running = x[i] + discount * running
            out[i] = running
        return out


# ==================== Curriculum Loss Weighting ====================

def get_loss_weights(update, total_updates):
    """3-phase curriculum: direction dominant early, PM catches up.

    Phase 1 (0 → 1/3):     Dir × 2.0, PM × 0.1  — direction learns first
    Phase 2 (1/3 → 2/3):   Dir × 2.0→1.0, PM × 0.1→0.5  — PM ramps up
    Phase 3 (2/3 → end):   Dir × 1.0→0.5, PM × 0.5→1.0  — balanced, PM takes over
    """
    p = update / max(total_updates - 1, 1)
    if p < 1/3:
        w_dir = 2.0
        w_pm = 0.1
    elif p < 2/3:
        t = (p - 1/3) * 3  # 0→1 within phase 2
        w_dir = 2.0 - 1.0 * t   # 2.0 → 1.0
        w_pm = 0.1 + 0.4 * t    # 0.1 → 0.5
    else:
        t = (p - 2/3) * 3  # 0→1 within phase 3
        w_dir = 1.0 - 0.5 * t   # 1.0 → 0.5
        w_pm = 0.5 + 0.5 * t    # 0.5 → 1.0
    return {'direction': w_dir, 'pm': w_pm}


# ==================== Main Vectorized PPO2 Trainer ====================
class VectorizedPPO2:
    """
    Vectorized PPO2 algorithm for discrete action spaces.
    
    Compatible with TradingNetworkLite and TradingEnv (dict observations).
    """
    def __init__(
        self,
        network: nn.Module,
        make_train_env,      # Function that creates a training environment
        make_val_env,        # Function that creates a validation environment
        make_test_env=None,  # Function that creates a test environment
        num_test_envs: int = 2,
        total_envs: int = 16,
        train_fraction: float = 0.8,
        steps_per_env: int = 512,
        val_steps_per_env: int = None,  # eval steps for val/test (default: same as train)
        total_timesteps: int = 100000,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gamma_direction: float = 0.97,  # L/S: shorter horizon (~33 min effective)
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 512,
        target_kl: float = 0.02,
        validation_freq: int = 5,
        checkpoint_freq: int = 50,
        early_stop_patience: int = 10,
        weight_decay: float = 0.0,
        load_model: bool = False,
        model_file: str = '',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = device
        self.total_envs = total_envs
        self.steps_per_env = steps_per_env
        self.val_steps_per_env = val_steps_per_env if val_steps_per_env is not None else steps_per_env
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.gamma_direction = gamma_direction
        self.gae_lambda = gae_lambda
        self.actor_gammas = {'direction': gamma_direction, 'pm': gamma}
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.initial_kl = 0.5  # Moderate initial KL — anneals to target_kl
        self.current_kl = self.initial_kl
        self.validation_freq = validation_freq
        self.checkpoint_freq = checkpoint_freq
        self.early_stop_patience = early_stop_patience
        
        # Calculate train/val split
        self.num_train_envs = int(total_envs * train_fraction)
        self.num_val_envs = total_envs - self.num_train_envs
        
        print(f"Environment allocation: {self.num_train_envs} training, {self.num_val_envs} validation")
        print(f"Train/Val split: {train_fraction*100:.0f}%/{100-train_fraction*100:.0f}%")
        
        # Create vectorized environments
        self.train_envs = SyncVectorEnv([lambda i=i: make_train_env(i) for i in range(self.num_train_envs)])
        self.val_envs = SyncVectorEnv([lambda i=i: make_val_env(i) for i in range(self.num_val_envs)])

        # Get observation shapes from a single environment
        single_env = make_train_env(0)
        self.obs_shapes = single_env.feature_shape
        self.obs_keys = list(self.obs_shapes.keys())
        single_env.close()
        
        print(f"Observation keys: {len(self.obs_keys)} features")
        
        # Initialize model
        self.network = network.to(device)
        self.base_lr = lr
        self.warmup_updates = 5  # Linear warmup over first 5 updates
        self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6
        )  # mode='max' — tracks val sharpe (higher is better)

        # Initialize buffers
        self.train_buffer = MultiActorPPOBuffer(
            self.num_train_envs, steps_per_env, self.obs_shapes, gamma, gae_lambda,
            actor_gammas=self.actor_gammas
        )
        self.val_buffer = MultiActorPPOBuffer(
            self.num_val_envs, self.val_steps_per_env, self.obs_shapes, gamma, gae_lambda,
            actor_gammas=self.actor_gammas
        )

        # Test envs (optional — for monitoring only, no training influence)
        self.num_test_envs = num_test_envs
        if make_test_env is not None:
            _make_test = make_test_env
            self.test_envs = SyncVectorEnv([lambda i=i: _make_test(i) for i in range(num_test_envs)])
            self.test_buffer = MultiActorPPOBuffer(
                num_test_envs, self.val_steps_per_env, self.obs_shapes, gamma, gae_lambda,
                actor_gammas=self.actor_gammas
            )
        else:
            self.test_envs = None
            self.test_buffer = None

        # Training statistics
        self.training_stats = {
            'train_losses': [],
            'train_returns': [],
            'train_lengths': [],
            'train_costs': [],
            'train_mdds': [],
            'val_losses': [],
            'val_returns': [],
            'val_lengths': [],
            'val_costs': [],
            'val_mdds': [],
            'best_val_loss': float('inf'),
            'best_metric': -float('inf'),       # for early stopping (-val_loss)
            'best_trading_metric': -float('inf'), # for model saving (composite)
            'no_improvement_count': 0,
            'update_steps': [],
        }

        # Load pretrained model if requested (after training_stats init)
        if load_model and model_file:
            self.load(f'models/{model_file}')

        # Data cycle tracking
        self._chunks_seen = set()       # cumulative set of chunk indices seen
        self._data_cycles = 0           # how many full cycles completed
        self._training_progress = 0.0   # 0 at start, 1 at end (for aux loss ramp)
        self._total_train_chunks = 0    # set during training

        # TensorBoard
        self.run_name = f'ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.writer = SummaryWriter(log_dir=f'runs/{self.run_name}')

        # Model directory for this run
        self.model_dir = f'models/{self.run_name}'
        os.makedirs(self.model_dir, exist_ok=True)

        # Text log file for easy reading
        self.log_path = f'runs/{self.run_name}/training.log'
        # Copy of full stdout output

        # Normalization parameters
        self.return_mean = 0.0
        self.return_std = 1.0

        # Reward normalization (per-actor running statistics)
        # Normalizes rewards by running std to stabilize value learning across
        # episodes with wildly different reward scales (trending vs choppy days)
        self.reward_rms = {name: RunningMeanStd() for name in ACTOR_NAMES}

        # Stateful rollout: maintain env state between training rollout collections
        # Prevents losing PnL/position/portfolio state at rollout boundaries
        self._train_obs = None
        self._last_train_actions = None
        self._val_obs = None
        self._last_val_actions = None
    
    def _obs_dict_to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert dict of numpy arrays to dict of tensors on device."""
        return {
            key: torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            for key, arr in obs_dict.items()
        }
    
    def collect_trajectories(self, envs: SyncVectorEnv, buffer: MultiActorPPOBuffer,
                            is_training: bool = True):
        """
        Collect fixed-length PPO rollouts from vectorized environments.
        
        Returns:
            avg_net, avg_mdd, avg_cost, avg_length
        """
        num_envs = envs.num_envs

        # Stateful rollouts: reuse cached env state for both train and val.
        # Avoids resetting envs between collections, preserving
        # PnL, position, and portfolio state across boundaries.
        if is_training:
            if self._train_obs is not None:
                current_obs = self._train_obs
                last_actions = self._last_train_actions
            else:
                current_obs, _ = envs.reset()
                last_actions = None
        else:
            if self._val_obs is not None:
                current_obs = self._val_obs
                last_actions = self._last_val_actions
            else:
                current_obs, _ = envs.reset()
                last_actions = None

        episode_pnl = []
        eff_clear, eff_buy, eff_sell = 0, 0, 0
        episode_pass_rates = []
        episode_lengths = []
        episode_costs = []
        env_lengths = np.zeros(num_envs, dtype=np.int32)
        last_infos = {}

        self.network.eval()

        n_steps = self.steps_per_env if is_training else self.val_steps_per_env
        for step in range(n_steps):
            obs_tensor = self._obs_dict_to_tensor(current_obs)

            with torch.no_grad():
                action_dict = self.network.get_action(
                    obs_tensor,
                    prev_actions=last_actions,
                    deterministic=not is_training
                )

            # Direction actor picks side (binary), PM provides sizing confidence + magnitude
            dir_actions_np = action_dict['direction_action'].cpu().numpy()
            pm_confid_np = action_dict['pm_confidence'].cpu().numpy()
            magnitude_np = action_dict['magnitude'].cpu().numpy()
            last_actions = action_dict['direction_action']

            actions = np.stack([
                dir_actions_np.astype(np.float32),
                pm_confid_np.astype(np.float32),
                magnitude_np.astype(np.float32),
            ], axis=-1)  # (num_envs, 3)
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)

            # Extract per-actor rewards from infos
            if isinstance(infos, dict):
                dir_base = np.asarray(infos.get('direction_reward', np.zeros(num_envs)), dtype=np.float32).flat[:num_envs].copy()
                signals = np.asarray(infos.get('signal', np.zeros(num_envs)), dtype=np.float32).flat[:num_envs].copy()
            else:
                dir_base = np.zeros(num_envs, dtype=np.float32)
                signals = np.zeros(num_envs, dtype=np.float32)

            # Track effective positions (clear/buy/sell based on actual position)
            positions = np.asarray(infos.get('position', np.zeros(num_envs)), dtype=np.float32).flat[:num_envs]
            eff_clear += (positions == 0).sum()
            eff_buy += (positions > 0).sum()
            eff_sell += (positions < 0).sum()

            # Part B: direction accuracy reward (binary: buy/sell always)
            # action 0=buy(+1), 1=sell(-1), signal is always ±1
            dir_actions = action_dict['direction_action'].cpu().numpy()
            dir_as_signal = np.where(dir_actions == 0, 1, -1)  # 0→+1(buy), 1→-1(sell)

            accuracy = np.zeros(num_envs, dtype=np.float32)
            accuracy += np.where(dir_as_signal == signals, 0.3, 0.0)    # correct side
            accuracy += np.where(dir_as_signal == -signals, -0.3, 0.0)  # wrong side

            dir_rewards = dir_base + accuracy
            pm_rewards = rewards  # main env reward → portfolio manager

            # Rescale raw rewards to ~unit variance
            dir_rewards = dir_rewards * 1
            pm_rewards = pm_rewards * 2

            # Replace NaN rewards with 0 (data NaN in expected_reward can poison one env)
            nan_dir = np.isnan(dir_rewards)
            nan_pm = np.isnan(pm_rewards)
            if nan_dir.any() or nan_pm.any():
                if not hasattr(self, '_rew_nan_warned'):
                    self._rew_nan_warned = True
                    print(f"      [NaN FIX] dir_rewards: {nan_dir.sum()}/{len(dir_rewards)} NaN zeroed, "
                          f"pm_rewards: {nan_pm.sum()}/{len(pm_rewards)} NaN zeroed")
                dir_rewards = np.where(nan_dir, 0.0, dir_rewards)
                pm_rewards = np.where(nan_pm, 0.0, pm_rewards)

            # Normalize rewards using running statistics for stable value learning
            if is_training:
                self.reward_rms['direction'].update(dir_rewards)
                self.reward_rms['pm'].update(pm_rewards)
            dir_rewards = np.clip(dir_rewards / self.reward_rms['direction'].std, -10, 10)
            pm_rewards = np.clip(pm_rewards / self.reward_rms['pm'].std, -10, 10)

            dir_val_np = action_dict['direction_value'].squeeze(-1).cpu().numpy()

            # Store all actor data
            actor_data = {
                            'direction': {
                                'act': action_dict['direction_action'].cpu().numpy(),
                                'logp': action_dict['direction_log_prob'].cpu().numpy(),
                                'rew': dir_rewards,
                                'val': dir_val_np,
                            },
                            'pm': {
                                'act': pm_confid_np,
                                'stop_act': action_dict['stop_ratio'].cpu().numpy(),
                                'logp': action_dict['pm_log_prob'].cpu().numpy(),
                                'stop_logp': action_dict['stop_log_prob'].cpu().numpy(),
                                'rew': pm_rewards,
                                'val': action_dict['pm_value'].squeeze(-1).cpu().numpy(),
                            },
                        }

            buffer.store(
                obs_dict=current_obs,
                actor_data=actor_data,
                done=dones.astype(np.float32),
            )

            env_lengths += 1

            # Handle episode completions — finish_path per episode
            done_env_indices = np.where(dones)[0]
            if len(done_env_indices) > 0:
                zero_vals = {name: np.zeros(num_envs) for name in ACTOR_NAMES}
                buffer.finish_path(zero_vals, env_indices=done_env_indices)

            for i in range(num_envs):
                if dones[i]:
                    info = self._extract_final_info(infos, i)

                    final_payout = info.get("payouts", 0.0)
                    final_cost = info.get("topstep_cost", 0.0)
                    final_pass_rate = info.get("exam_pass_rate", 0.0)

                    episode_pnl.append(final_payout)
                    episode_pass_rates.append(final_pass_rate)
                    episode_costs.append(final_cost)
                    episode_lengths.append(env_lengths[i])
                    env_lengths[i] = 0

            current_obs = next_obs
            last_infos = infos

            # Progress update every 10K steps
            total_steps = (step + 1) * num_envs
            if is_training and total_steps % 5000 < num_envs and step > 0:
                if episode_pnl:
                    avg_net = np.mean(episode_pnl)
                    avg_cost = np.mean(episode_costs)
                    avg_pass = np.mean(episode_pass_rates)
                else:
                    # No episodes completed yet — use partial info from current step
                    avg_net = float(np.mean(infos.get('payouts', np.zeros(num_envs))))
                    avg_cost = float(np.mean(infos.get('topstep_cost', np.zeros(num_envs))))
                    avg_pass = float(np.mean(infos.get('exam_pass_rate', np.zeros(num_envs))))
                avg_acc_pnl = float(np.mean(infos.get('acc_pnl', np.zeros(num_envs))))
                avg_exam_counts = float(np.mean(infos.get('exam_counts', np.zeros(num_envs))))
                avg_comm = float(np.mean(infos.get('total_commission', np.zeros(num_envs))))
                n_eps = len(episode_pnl)
                # Track data cycles cumulatively
                for env in envs.envs:
                    self._chunks_seen.add(env.played_chunk_idx)
                if self._total_train_chunks > 0 and len(self._chunks_seen) >= self._total_train_chunks:
                    self._data_cycles += 1
                    self._chunks_seen.clear()
                print(f"    step {step+1:>4d}/{n_steps} ({total_steps:>6,} total) | "
                      f"ep_reset:{n_eps:<2d} | cycle:{self._data_cycles} | AccPnl:{avg_acc_pnl:>+7.0f} | Payout:{avg_net:>+7.0f} | PassRate:{avg_pass:.2f} | ExamCost:{avg_cost:>+7.0f} | ExamsCnt:{avg_exam_counts:.1f} | Comm:{avg_comm:+.0f}")

        # Bootstrap values for unfinished episodes
        with torch.no_grad():
            obs_tensor = self._obs_dict_to_tensor(current_obs)
            action_dict = self.network.get_action(obs_tensor, deterministic=True)

        last_vals = {
            'direction': action_dict['direction_value'].squeeze(-1).cpu().numpy(),
            'pm': action_dict['pm_value'].squeeze(-1).cpu().numpy(),
        }
        buffer.finish_path(last_vals)
        
        # Compute metrics
        num_completed = len(episode_pnl)
        if num_completed > 0:
            avg_payout = float(np.mean(episode_pnl))
            avg_pass_rate = float(np.mean(episode_pass_rates))
            avg_cost = float(np.mean(episode_costs))
            avg_length = float(np.mean(episode_lengths))
        else:
            # Get partial stats from last step's infos
            if isinstance(last_infos, dict):
                partial_payout = last_infos.get('payouts', np.zeros(num_envs))
                partial_pass_rate = last_infos.get('exam_pass_rate', np.zeros(num_envs))
                partial_costs = last_infos.get('topstep_cost', np.zeros(num_envs))
                avg_payout = float(np.mean(partial_payout))
                avg_pass_rate = float(np.mean(partial_pass_rate))
                avg_cost = float(np.mean(partial_costs))
            else:
                avg_payout = 0.0
                avg_pass_rate = 0.0
                avg_cost = 0.0
            avg_length = float(self.steps_per_env)

        # Avg PnL / Agg Sum + acc_pnl + stop counts from live envs
        avg_nets, total_nets, acc_pnls, sw_counts, sl_counts, comm_totals = [], [], [], [], [], []
        for env in envs.envs:
            ap, ag = env.get_final_metrics()
            avg_nets.append(ap)
            total_nets.append(ag)
            acc_pnls.append(env.pm.topstep.acc_pnl)
            sw_counts.append(env.stop_win_count)
            sl_counts.append(env.stop_loss_count)
            comm_totals.append(env.pm.topstep.total_commission)
        avg_sw_cnt = float(np.mean(sw_counts))
        avg_sl_cnt = float(np.mean(sl_counts))
        avg_comm = float(np.mean(comm_totals))

        avg_net = float(np.mean(avg_nets))
        total_net = float(np.mean(total_nets))
        avg_acc_pnl = float(np.mean(acc_pnls))

        # Cache env state for next rollout (stateful collection for both train and val)
        if is_training:
            self._train_obs = current_obs
            self._last_train_actions = last_actions
        else:
            self._val_obs = current_obs
            self._last_val_actions = last_actions

        eff_total = max(eff_clear + eff_buy + eff_sell, 1)
        eff_pcts = (eff_clear / eff_total * 100, eff_buy / eff_total * 100, eff_sell / eff_total * 100)
        return avg_payout, avg_pass_rate, avg_cost, avg_length, avg_net, total_net, avg_acc_pnl, avg_sw_cnt, avg_sl_cnt, avg_comm, eff_pcts
    
    def _extract_final_info(self, infos, env_idx: int) -> dict:
        """Extract final episode info from vectorized env infos."""
        if isinstance(infos, dict):
            if 'final_info' in infos:
                final_info_array = infos['final_info']
                if isinstance(final_info_array, np.ndarray) and env_idx < len(final_info_array):
                    return final_info_array[env_idx] if final_info_array[env_idx] is not None else {}
            return infos
        elif isinstance(infos, (list, tuple)) and env_idx < len(infos):
            return infos[env_idx]
        return {}
    
    @staticmethod
    def _composite_metric(acc_pnl, exam_cost, payout):
        """Composite validation metric: AccPnl * 0.45 + ExamCost + Payout.
        Higher is better. Directly reflects trading profitability on val."""
        a = acc_pnl if not np.isnan(acc_pnl) else 0.0
        c = exam_cost if not np.isnan(exam_cost) else 0.0
        p = payout if not np.isnan(payout) else 0.0
        return a * 0.45 + c + p  # c is already negative (cost), naturally penalizes

    def _adapt_value_grad_scale(self, value_loss, policy_loss):
        """Auto-tune value_grad_scale based on value/policy loss ratio.
        High value_loss relative to policy → scale up (value needs more gradient).
        Low value_loss → scale down (value is fine, reduce interference).
        Clamps [0.1, 0.5], EMA smoothed."""
        if policy_loss < 1e-8:
            return
        ratio = value_loss / (policy_loss + value_loss)
        # Center at 0.3 when ratio=0.35, adjust ±0.1
        target = 0.3 + 0.2 * (ratio - 0.35)
        target = max(0.1, min(0.5, target))
        current = self.network.value_grad_scale
        self.network.value_grad_scale = current * 0.99 + target * 0.01

    def compute_loss(self, data: Dict, is_training: bool = True,
                     actor_weights: Optional[Dict] = None):
        """Compute PPO2 clipped loss across both actors."""
        obs_dict = {
            key: torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            for key, arr in data['obs'].items()
        }

        # Build actions dict for evaluate_action
        actions_dict = {
            'direction_action': torch.as_tensor(data['direction_act'], dtype=torch.long, device=self.device),
            'pm_confidence': torch.as_tensor(data['pm_act'], dtype=torch.float32, device=self.device),
            'stop_ratio': torch.as_tensor(data['pm_stop_act'], dtype=torch.float32, device=self.device),
        }

        eval_dict = self.network.evaluate_action(obs_dict, actions_dict)

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        per_actor_policy = {}
        per_actor_value = {}

        # Curriculum actor weights (default fallback for backward compat)
        if actor_weights is None:
            actor_weights = {'direction': 0.5, 'pm': 1.0}

        for name in ACTOR_NAMES:
            old_logp = torch.as_tensor(data[f'{name}_logp'], dtype=torch.float32, device=self.device)
            returns = torch.as_tensor(data[f'{name}_ret'], dtype=torch.float32, device=self.device)
            advantages = torch.as_tensor(data[f'{name}_adv'], dtype=torch.float32, device=self.device)
            old_values = torch.as_tensor(data[f'{name}_val'], dtype=torch.float32, device=self.device)

            log_probs = eval_dict[f'{name}_log_prob']
            values = eval_dict[f'{name}_value'].squeeze(-1)
            entropy = eval_dict[f'{name}_entropy']

            # PM: joint log_prob = pm_log_prob + stop_log_prob
            if name == 'pm':
                old_stop_logp = torch.as_tensor(data['pm_stop_logp'], dtype=torch.float32, device=self.device)
                log_probs = log_probs + eval_dict['stop_log_prob']
                old_logp = old_logp + old_stop_logp
                entropy = entropy + eval_dict['stop_entropy']

            # DEBUG: trace NaN source (remove after fix) — prints once per update
            has_nan = False
            for label, t in [('old_logp', old_logp), ('returns', returns), ('advantages', advantages),
                             ('old_values', old_values), ('log_probs', log_probs), ('values', values)]:
                if torch.isnan(t).any() or torch.isinf(t).any():
                    has_nan = True
                    nan_c = torch.isnan(t).sum().item()
                    inf_c = torch.isinf(t).sum().item()
                    valid = t[~torch.isnan(t) & ~torch.isinf(t)]
                    if len(valid) > 0:
                        print(f"      [NaN TRACE] {name}.{label}: {nan_c} NaN, {inf_c} Inf / {t.numel()} total, "
                              f"valid min={valid.min().item():.4f} max={valid.max().item():.4f}")
                    else:
                        print(f"      [NaN TRACE] {name}.{label}: ALL NaN/Inf ({t.numel()} elements)")
            if has_nan:
                break  # only trace first bad batch per update

            # Clipped surrogate
            ratio = torch.exp(log_probs - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss
            v_clipped = old_values + torch.clamp(values - old_values, -self.clip_epsilon, self.clip_epsilon)
            vl_unclipped = (values - returns) ** 2
            vl_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

            w = actor_weights[name]
            total_policy_loss = total_policy_loss + w * policy_loss
            total_value_loss = total_value_loss + value_loss  # unweighted — value heads always need full gradient
            total_entropy = total_entropy + entropy.mean()
            per_actor_policy[name] = policy_loss.item()
            per_actor_value[name] = value_loss.item()

        # L1 regularization on first-contact layer weights
        l1_loss = self.network.l1_first_layer_loss()

        loss = (total_policy_loss + self.value_coef * total_value_loss
                - self.entropy_coef * (total_entropy / 2.0)
                + self.network.l1_coef * l1_loss)

        # KL early stopping — both actors (2nd-order approximation, always ≥ 0)
        dir_old_logp = torch.as_tensor(data['direction_logp'], dtype=torch.float32, device=self.device)
        pm_old_logp = torch.as_tensor(data['pm_logp'], dtype=torch.float32, device=self.device)
        pm_old_stop_logp = torch.as_tensor(data['pm_stop_logp'], dtype=torch.float32, device=self.device)
        joint_old = dir_old_logp + pm_old_logp + pm_old_stop_logp
        joint_new = eval_dict['direction_log_prob'] + eval_dict['pm_log_prob'] + eval_dict['stop_log_prob']
        log_ratio = joint_new - joint_old
        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

        return loss, approx_kl, total_policy_loss.item(), total_value_loss.item(), (total_entropy / 2.0).item(), l1_loss.item(), per_actor_policy, per_actor_value
    
    def train_epoch(self, train_data: Dict,
                    actor_weights: Optional[Dict] = None):
        """Train for one epoch on collected data."""
        total_loss = 0
        total_kl = 0
        batch_count = 0
        last_kl = 0
        l1_loss = 0.0
        actor_losses = {'direction': 0.0, 'pm': 0.0}
        actor_val_losses = {'direction': 0.0, 'pm': 0.0}

        num_samples = len(train_data['pm_act'])

        for epoch in range(self.n_epochs):
            batch_size = min(self.batch_size, num_samples)

            # Shuffle to break temporal correlations
            indices = np.random.permutation(num_samples)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                # Create batch with all actor fields
                batch = {
                    'obs': {key: arr[idx] for key, arr in train_data['obs'].items()},
                }
                for name in ACTOR_NAMES:
                    for field in ('act', 'logp', 'ret', 'val'):
                        batch[f'{name}_{field}'] = train_data[f'{name}_{field}'][idx]
                    # Per-mini-batch advantage normalization
                    # Skip for tiny tail batches (<64 samples) — noisy statistics
                    adv = train_data[f'{name}_adv'][idx]
                    if len(adv) >= 64:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    batch[f'{name}_adv'] = adv
                batch['pm_stop_act'] = train_data['pm_stop_act'][idx]
                batch['pm_stop_logp'] = train_data['pm_stop_logp'][idx]

                loss, approx_kl, policy_loss, value_loss, entropy, l1_loss, actor_losses, actor_val_losses = self.compute_loss(
                    batch, is_training=True, actor_weights=actor_weights
                )
                last_kl = approx_kl

                # KL check BEFORE gradient step — prevents any batch from causing policy collapse
                if approx_kl > 1.5 * self.current_kl:
                    print(f"      Early stop at epoch {epoch}: KL={approx_kl:.4f} > {1.5*self.current_kl:.4f}")
                    break

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"      WARNING: NaN/Inf loss at epoch {epoch} | "
                          f"policy:{policy_loss:.4f} value:{value_loss:.4f} entropy:{entropy:.4f} l1:{l1_loss:.4f}")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                # Check for NaN gradients before stepping — prevents weight corruption
                grad_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in self.network.parameters())
                if grad_nan:
                    print(f"      WARNING: NaN gradients at epoch {epoch}, skipping update")
                    self.optimizer.zero_grad()
                    continue
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self._adapt_value_grad_scale(value_loss, policy_loss)

                total_loss += loss.item()
                total_kl += approx_kl
                batch_count += 1

            if last_kl > 1.5 * self.current_kl:
                break

        avg_loss = total_loss / max(batch_count, 1)
        avg_kl = total_kl / max(batch_count, 1)

        return avg_loss, avg_kl, policy_loss, value_loss, entropy, last_kl, l1_loss, actor_losses, actor_val_losses
    
    def evaluate_test(self):
        """Run test set evaluation (monitoring only — no training influence)."""
        if self.test_envs is None or self.test_buffer is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        test_buffer = self.test_buffer
        test_envs = self.test_envs
        test_buffer.ptr = 0
        test_buffer.path_start_idx = np.zeros(self.num_test_envs, dtype=np.int32)
        test_payout, test_pass_rate, test_cost, _, test_avg_net, test_total_net, test_acc_pnl, test_sw, test_sl, test_comm, test_eff_pcts = self.collect_trajectories(
            test_envs, test_buffer, is_training=False
        )
        test_exam_counts = float(np.mean([e.pm.topstep.exam_counts for e in test_envs.envs]))
        # Action distribution
        test_data = test_buffer.get_data_for_envs(np.arange(self.num_test_envs))
        acts = test_data['direction_act']
        if isinstance(acts, torch.Tensor):
            acts = acts.cpu().numpy()
        n = len(acts)
        pct_buy = (acts == 0).sum() / n * 100   # 0=buy
        pct_sell = (acts == 1).sum() / n * 100  # 1=sell
        test_buffer.clear_all_data(np.arange(self.num_test_envs))
        return test_payout, test_pass_rate, test_cost, test_avg_net, test_total_net, test_exam_counts, pct_buy, pct_sell, test_acc_pnl

    def validate(self):
        """Run validation on validation environments."""
        # Reset validation buffer
        self.val_buffer.ptr = 0
        self.val_buffer.path_start_idx = np.zeros(self.num_val_envs, dtype=np.int32)

        # Collect validation data
        val_payout, val_pass_rate, val_cost, val_length, val_avg_net, val_total_net, val_acc_pnl, val_sw, val_sl, val_comm, val_eff_pcts = self.collect_trajectories(
            self.val_envs, self.val_buffer, is_training=False
        )
        val_exam_counts = float(np.mean([e.pm.topstep.exam_counts for e in self.val_envs.envs]))

        # Get validation data
        val_data = self.val_buffer.get_data_for_envs(np.arange(self.num_val_envs))

        # Action distribution
        val_acts = val_data['direction_act']
        if isinstance(val_acts, torch.Tensor):
            val_acts = val_acts.cpu().numpy()
        n_val = len(val_acts)
        val_pct_buy = (val_acts == 0).sum() / n_val * 100   # 0=buy
        val_pct_sell = (val_acts == 1).sum() / n_val * 100  # 1=sell

        # Normalize advantages (same per-batch treatment as training)
        for name in ACTOR_NAMES:
            adv = val_data[f'{name}_adv']
            val_data[f'{name}_adv'] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Clear buffer
        self.val_buffer.clear_all_data(np.arange(self.num_val_envs))
        
        # Compute validation loss
        with torch.no_grad():
            val_loss, _, _, _, _, _, _, _ = self.compute_loss(val_data, is_training=False)
        
        # Store statistics
        self.training_stats['val_returns'].append(val_payout)
        self.training_stats['val_lengths'].append(val_length)
        self.training_stats['val_costs'].append(val_cost)
        self.training_stats['val_mdds'].append(val_pass_rate)
        self.training_stats['val_losses'].append(val_loss.item())

        return val_loss.item(), val_payout, val_pass_rate, val_length, val_cost, val_avg_net, val_total_net, val_exam_counts, val_pct_buy, val_pct_sell, val_acc_pnl, val_sw, val_sl, val_comm, val_eff_pcts
    
    def train(self) -> nn.Module:
        """Main training loop with validation and early stopping."""
        num_updates = self.total_timesteps // (self.num_train_envs * self.steps_per_env)
        
        print(f"\n{'='*70}")
        print(f"Starting Vectorized PPO2 Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Training envs: {self.num_train_envs} | Validation envs: {self.num_val_envs}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Updates: {num_updates}")
        print(f"Network params: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"{'='*70}\n")
        
        start_time = time.time()

        # Initialize env state for stateful rollout collection
        self._train_obs, _ = self.train_envs.reset()
        self._last_train_actions = None
        self._total_train_chunks = len(self.train_envs.envs[0].data_pool)

        for update in range(num_updates):
            self._training_progress = (update + 1) / num_updates  # 0→1
            self.training_stats['update_steps'].append(update)

            # Anneal KL threshold: linear decay from initial_kl to target_kl
            progress = update / max(num_updates - 1, 1)
            self.current_kl = self.initial_kl + (self.target_kl - self.initial_kl) * progress

            # Linear LR warmup for first N updates — also relax KL during warmup
            if update < self.warmup_updates:
                warmup_lr = self.base_lr * (update + 1) / self.warmup_updates
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr
                # Untrained network causes large KL on first updates; low LR controls step size
                self.current_kl = 5.0

            # Reset training buffer
            self.train_buffer.ptr = 0
            self.train_buffer.path_start_idx = np.zeros(self.num_train_envs, dtype=np.int32)
            
            # Collect training trajectories
            train_payout, train_pass_rate, train_cost, train_length, train_avg_net, train_total_net, train_acc_pnl, train_sw, train_sl, train_comm, train_eff_pcts = self.collect_trajectories(
                self.train_envs, self.train_buffer, is_training=True
            )
            train_exam_counts = float(np.mean([e.pm.topstep.exam_counts for e in self.train_envs.envs]))

            # Store training stats
            self.training_stats['train_returns'].append(train_payout)
            self.training_stats['train_mdds'].append(train_pass_rate)
            self.training_stats['train_lengths'].append(train_length)
            self.training_stats['train_costs'].append(train_cost)
            
            # Get training data
            train_data = self.train_buffer.get_data_for_envs(np.arange(self.num_train_envs))
            
            # NOTE: Do NOT normalize returns here. Advantages are already
            # normalized in the buffer (get_data_for_envs). Normalizing returns
            # as well creates a moving target for the value head — the value
            # function learns to predict normalized returns, but the normalization
            # statistics shift each rollout, destabilizing value learning.
            
            # Clear buffer
            self.train_buffer.clear_all_data(np.arange(self.num_train_envs))
            
            # Compute action distribution (direction actions)
            actions_taken = train_data['direction_act']
            if isinstance(actions_taken, torch.Tensor):
                actions_taken = actions_taken.cpu().numpy()
            n_total = max(len(actions_taken), 1)
            n_buy = (actions_taken == 0).sum()   # 0=buy
            n_sell = (actions_taken == 1).sum()   # 1=sell
            pct_buy = (n_buy / n_total) * 100
            pct_sell = (n_sell / n_total) * 100

            self.network.train()

            # Curriculum loss weights (automatic 3-phase schedule)
            curr_weights = get_loss_weights(update, num_updates)
            avg_loss, avg_kl, policy_loss, value_loss, entropy, last_kl, l1_loss, actor_losses, actor_val_losses = self.train_epoch(
                train_data, actor_weights=curr_weights
            )
            self.training_stats['train_losses'].append(avg_loss)

            # TensorBoard: all metrics
            self.writer.add_scalar('curriculum/w_dir', curr_weights['direction'], update)
            self.writer.add_scalar('curriculum/w_pm', curr_weights['pm'], update)
            self.writer.add_scalar('train/action_buy_pct', pct_buy, update)
            self.writer.add_scalar('train/action_sell_pct', pct_sell, update)
            self.writer.add_scalar('train/loss', avg_loss, update)
            self.writer.add_scalar('train/dir_policy_loss', actor_losses['direction'], update)
            self.writer.add_scalar('train/pm_policy_loss', actor_losses['pm'], update)
            self.writer.add_scalar('train/dir_value_loss', actor_val_losses['direction'], update)
            self.writer.add_scalar('train/pm_value_loss', actor_val_losses['pm'], update)
            self.writer.add_scalar('train/l1_loss', l1_loss, update)
            self.writer.add_scalar('train/entropy', entropy, update)
            self.writer.add_scalar('train/kl', last_kl, update)
            self.writer.add_scalar('train/payout', train_payout, update)
            self.writer.add_scalar('train/cost', train_cost, update)
            self.writer.add_scalar('train/pass_rate', train_pass_rate, update)
            self.writer.add_scalar('train/avg_net', train_avg_net, update)
            self.writer.add_scalar('train/total_net', train_total_net, update)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], update)
            self.writer.add_scalar('train/value_grad_scale', self.network.value_grad_scale, update)

            # Input gate values (if enabled)
            if self.network.use_input_gates:
                gate_vals = self.network.get_gate_values()
                for gate_name, gate_tensor in gate_vals.items():
                    self.writer.add_scalar(f'gates/{gate_name}_mean', gate_tensor.mean().item(), update)
                    self.writer.add_scalar(f'gates/{gate_name}_min', gate_tensor.min().item(), update)

            # Validation + Test monitoring
            val_str = ""
            if (update > 0 and update % self.validation_freq == 0) or update == num_updates - 1:
                val_loss, val_payout, val_pass_rate, val_length, val_cost, val_avg_net, val_total_net, val_exam_counts, val_pct_buy, val_pct_sell, val_acc_pnl, val_sw, val_sl, val_comm, val_eff_pcts = self.validate()

                # TensorBoard: validation metrics
                self.writer.add_scalar('val/loss', val_loss, update)
                self.writer.add_scalar('val/payout', val_payout, update)
                self.writer.add_scalar('val/cost', val_cost, update)
                self.writer.add_scalar('val/pass_rate', val_pass_rate, update)

                val_metric = self._composite_metric(val_acc_pnl, val_cost, val_payout)

                # Update LR scheduler on composite metric
                self.lr_scheduler.step(val_metric)

                # Early stopping uses composite metric once agent trades, else val loss
                warmup_updates = num_updates // 3
                saved = ""

                # Model saving: use composite trading metric (independent of early stopping)
                if val_metric > self.training_stats.get('best_trading_metric', -float('inf')):
                    self.training_stats['best_trading_metric'] = val_metric
                    self.save(f"{self.model_dir}/best_trading_model.pth", update=update)
                    saved = " *SAVED*"

                # Early stopping: use -val_loss (skip during warmup)
                if update >= warmup_updates and not np.isnan(val_loss):
                    es_metric = -val_loss  # lower loss = higher metric = improvement
                    if es_metric > self.training_stats.get('best_metric', -float('inf')):
                        self.training_stats['best_metric'] = es_metric
                        self.training_stats['no_improvement_count'] = 0
                    else:
                        self.training_stats['no_improvement_count'] += 1

                if val_loss < self.training_stats['best_val_loss']:
                    self.training_stats['best_val_loss'] = val_loss

                self.writer.add_scalar('val/avg_net', val_avg_net if not np.isnan(val_avg_net) else 0.0, update)
                self.writer.add_scalar('val/total_net', val_total_net if not np.isnan(val_total_net) else 0.0, update)
                self.writer.add_scalar('val/metric', val_metric, update)
                val_str = (f"\n  VAL  | Payout:{val_payout:+.0f} | AccPnl:{val_acc_pnl:+.0f} | PassRate:{val_pass_rate:.2f} | ExamCost:{val_cost:.0f} | ExamsCnt:{val_exam_counts:.1f} | AvgNet:{val_avg_net:+.0f} | TotalNet:{val_total_net:+.0f} | "
                           f"Loss:{val_loss:.4f} | SW:{val_sw:.0f} SL:{val_sl:.0f} Comm:{val_comm:+.0f} | Clear:{val_eff_pcts[0]:4.1f}% Buy:{val_eff_pcts[1]:4.1f}% Sell:{val_eff_pcts[2]:4.1f}%{saved}")

            # Consistent one-line log for EVERY update
            elapsed = time.time() - start_time
            log_line = (f"[{update+1:3d}/{num_updates}] "
                  f"Payout:{train_payout:+.0f} | AccPnl:{train_acc_pnl:+.0f} | PassRate:{train_pass_rate:.2f} | ExamCost:{train_cost:.0f} | ExamsCnt:{train_exam_counts:.1f} | AvgNet:{train_avg_net:+.0f} | TotalNet:{train_total_net:+.0f} | "
                  f"Loss:{avg_loss:.4f} | DirLoss:{actor_losses['direction']:.4f} PmLoss:{actor_losses['pm']:.4f} | KL:{last_kl:+.4f}(lim:{1.5*self.current_kl:.2f}) | "
                  f"SW:{train_sw:.0f} SL:{train_sl:.0f} Comm:{train_comm:+.0f} | Clear:{train_eff_pcts[0]:4.1f}% Buy:{train_eff_pcts[1]:4.1f}% Sell:{train_eff_pcts[2]:4.1f}% | "
                  f"LR:{self.optimizer.param_groups[0]['lr']:.1e} | {elapsed:.0f}s"
                  f"{val_str}")
            print(f"\n{log_line}", flush=True)
            with open(self.log_path, 'a') as f:
                f.write(log_line + '\n')

            # Periodic checkpoint
            if self.checkpoint_freq > 0 and (update + 1) % self.checkpoint_freq == 0:
                ckpt_path = f"{self.model_dir}/checkpoint_update_{update+1}.pth"
                self.save(ckpt_path, update=update)
                print(f"  >> Checkpoint saved: {ckpt_path}")

            # Early stopping
            if self.training_stats.get('no_improvement_count', 0) >= self.early_stop_patience:
                print(f"\nEarly stopping after {update+1} updates (no improvement for {self.early_stop_patience})")
                break
        
        # Cleanup
        self.writer.close()
        self.train_envs.close()
        self.val_envs.close()
        if self.test_envs is not None:
            self.test_envs.close()
        
        # Load best model
        try:
            self.load(f"{self.model_dir}/best_trading_model.pth")
            print("Loaded best model for final evaluation")
        except Exception as e:
            print(f"Could not load best model: {e}")
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best Val Loss: {self.training_stats['best_val_loss']:.4f}")
        print(f"Best Trading Metric: {self.training_stats.get('best_trading_metric', 0.0):.4f}")
        print(f"Total Time: {time.time() - start_time:.1f}s")
        print(f"{'='*70}")
        
        return self.network
    
    def save(self, path: str, update: int = -1):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'obs_shapes': self.obs_shapes,
            'update': update,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        missing, unexpected = self.network.load_state_dict(checkpoint['network_state_dict'], strict=False)
        if missing:
            print(f"  Missing keys (initialized fresh): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_stats' in checkpoint:
            self.training_stats.update(checkpoint['training_stats'])
        print(f"Model loaded from {path}")


# ==================== Data Chunking Utilities ====================

def create_episodes(df, steps_per_episode):
    episodes = []
    current_episode_days = []
    current_steps = 0
    
    for _, row in df.iterrows():
        current_steps += row['count']
        current_episode_days.append((row['first_index'],row['last_index']))
        
        if current_steps >= steps_per_episode:
            episodes.append(current_episode_days)
            current_episode_days = []
            current_steps = 0
    
    # If there are remaining days that don't form a full episode, we can either discard them or add them to the last episode.
    if current_episode_days:
        # If we have already created at least one episode, we can add the remaining days to the last episode.
        if episodes:
            episodes[-1].extend(current_episode_days)
        else:
            # If we haven't created any episode, then we have only one episode (even if it's short).
            episodes.append(current_episode_days)
    
    return episodes

def create_date_chunk_info(df):
    import datetime
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = (df['ts'] - datetime.timedelta(hours=5)).dt.date 

    tem = df.groupby('date').agg(
        count=('index', 'count'),
        first_index=('index', 'first'),
        last_index=('index', 'last'),
    ).reset_index()

    tem = tem[tem['count']>500]
    tem.reset_index(drop=True)

    tem.to_excel('./data/sample/date_chunk_info.xlsx')

    df.drop(columns=['date'],inplace=True)
    return tem


def create_chunks(df,steps_per_episode=None,minimum_steps_per_episode=None,max_steps_per_episode=None,is_daily=False):
    import pandas as pd
    if is_daily:
        chunk_info = create_date_chunk_info(df)
    else:
        chunk_info = pd.read_pickle('./chunk_info.pkl')

    WINDOW = 20
    res = create_episodes(chunk_info, steps_per_episode)
    res = [(max(item[0][0]-1,0),item[-1][-1]) for item in res]
    res = res[1:] #FIXME why skip first chunk
    all_chunks = [ df.loc[int(item[0]-WINDOW):int(item[1])] for item in res if item[1]+WINDOW-item[0]>minimum_steps_per_episode]
    if max_steps_per_episode is not None:
        all_chunks = [chunk.iloc[:max_steps_per_episode] for chunk in all_chunks]
        
    return all_chunks

if __name__ == "__main__":
    import pandas as pd
    from env import TradingEnv
    from network import TradingNetworkLite
    from post_preprocessing import convert_to_numpy
    
    # Check if CUDA available
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pre-scaled data
    try:
        data = pd.read_pickle('./data/final_data.pkl')
        print(f"Loaded data: {len(data):,} rows")
    except FileNotFoundError:
        print("Error: final_data.pkl not found. Run scripts/prepare_final_data.py first.")
        exit(1)
    
    # # CHUNK SIZE CALCULATION (~3M data)
    # # For ~3M rows at 1-min bars (~2 years of trading data):
    # - CHUNK_SIZE = 8192 → ~366 chunks (~8 trading days per episode)
    # - CHUNK_SIZE = 6144 → ~488 chunks (~6 trading days per episode)  
    # - CHUNK_SIZE = 4096 → ~732 chunks (~4 trading days per episode)
    #
    # Recommendation: 8192 for learning multi-day patterns
    # # #normal
    STEPS_PER_EPISODE  = 30 * 1020  # 30 trading days per episode (~30,600 bars)
    MIN_STEPS_PER_EPISODE = 25 * 1020  # at least 25 days
    MAX_STEPS_PER_EPISODE = 32 * 1020
    is_daily= False

    #daily
    # STEPS_PER_EPISODE  = 900   # ~8 trading days per episode 
    # MIN_STEPS_PER_EPISODE = 900
    # MAX_STEPS_PER_EPISODE = 1000
    # is_daily= True

    MODEL_FILE = ''

    all_chunks_df = create_chunks(data, steps_per_episode=STEPS_PER_EPISODE, minimum_steps_per_episode=MIN_STEPS_PER_EPISODE, max_steps_per_episode=MAX_STEPS_PER_EPISODE,is_daily=is_daily)
    # Convert DataFrame chunks to numpy dicts for env
    all_chunks = [convert_to_numpy(chunk.reset_index(drop=True)) for chunk in all_chunks_df]
    n_chunks = len(all_chunks)
    print(f"Created {n_chunks} chunks of ~{STEPS_PER_EPISODE} steps each")

    # Split chunks into train/val/test — 1 test, 3 val, rest train
    test_chunks = all_chunks[-1:]
    val_chunks = all_chunks[-4:-1]
    train_chunks = all_chunks[:-4]

    print(f"Training chunks: {len(train_chunks)}")
    print(f"Validation chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)} [held out]")

    data_size = sum(len(chunk['ohlc']) for chunk in train_chunks)
    print(f"Training data size: {data_size:,} rows")

    # Create network
    network = TradingNetworkLite(num_actions=2,
                                l1_coef=1e-5, use_input_gates=True)
    print(f"Network created: {sum(p.numel() for p in network.parameters()):,} parameters")
    
    # # PPO2 CONFIGURATION FOR ~3M DATA
    # # With 16 envs (12 train @ 0.75) and steps_per_env=512:
    # - Each update: 12 * 512 = 6,144 timesteps
    # - 3M * 3 epochs / 6,144 = ~1,465 updates
    # - Validation every 10 = ~146 validations

    config = {
                'network': network,
                'make_train_env': lambda i=0: TradingEnv(data=train_chunks[i % len(train_chunks)], data_pool=train_chunks, env_id=i, num_envs=10, obs_noise_std=0.005),
                'make_val_env': lambda i=0: TradingEnv(data=val_chunks[i % len(val_chunks)], data_pool=val_chunks, env_id=i, num_envs=3),
                'make_test_env': lambda i=0: TradingEnv(data=test_chunks[i % len(test_chunks)], data_pool=test_chunks, env_id=i, num_envs=1),
                'num_test_envs': 1,

                # Environment settings
                'total_envs': 13,           # 10 train, 3 val
                'train_fraction': 10/13,    # 3 val envs, 10 train
                'steps_per_env': 2048,        # collect 2048 steps then train — frequent updates
                'val_steps_per_env': 2048,    # 2-day sliding window (stateful, advances each check)

                # Training schedule
                'total_timesteps': data_size * 2,  # shorter run for testing

                # PPO hyperparameters (tuned for large dataset)
                'lr': 1e-4,                 # Lower LR for stable convergence
                'gamma': 0.99,              # PM discount (session-level return horizon)
                'gamma_direction': 0.97,    # L/S discount (shorter horizon ~33 min for signal quality)
                'gae_lambda': 0.95,         # GAE lambda
                'clip_epsilon': 0.1,        # Tighter clip for more stable policy updates
                'value_coef': 0.5,          # Value loss weight
                'entropy_coef': 0.01,       # Run 5 value
                'max_grad_norm': 0.5,       # Gradient clipping

                # Optimization
                'n_epochs': 10,             # Run 5 value
                'batch_size': 1024*2,       # Larger batch = more stable
                'target_kl': 0.02,          # Standard PPO KL target

                # Validation & early stopping
                'validation_freq': 5,       # Run 5 value
                'checkpoint_freq': 10,      # Save checkpoint every 10 updates
                'early_stop_patience': 999, # Disabled — diagnostic run with clean data

                # Resume training from checkpoint
                'load_model': False,       # Fresh start
                'model_file': '',

                # Regularization
                'weight_decay': 1e-4,       # Original value — 5e-4 over-regularized
                'device': device,
            }
    
    print("\nPPO2 Configuration:")
    print("=" * 50)
    for key, value in config.items():
        if key not in ['network', 'make_train_env', 'make_val_env', 'make_test_env']:
            print(f"{key:20}: {value}")
    print("=" * 50)
    
    # Create and train agent
    agent = VectorizedPPO2(**config)

    # Save initial (random) model for architecture analysis
    init_path = f"{agent.model_dir}/initial_model.pth"
    agent.save(init_path, update=-1)
    print(f"Saved initial model: {init_path}")

    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trained_model = agent.train()
    
    # Save final model
    agent.save(f"{agent.model_dir}/final_trading_model.pth")

    print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved as '{agent.model_dir}/final_trading_model.pth'")

    # FULL TEST EVALUATION — run best model on entire test chunk (~30 days)
    print(f"\n{'='*70}")
    print(f"FULL TEST EVALUATION (best model, ~{len(test_chunks[0]['ohlc'])} steps)")
    print(f"{'='*70}")
    best_path = f'{agent.model_dir}/best_trading_model.pth'
    if os.path.exists(best_path):
        # Load best model
        checkpoint = torch.load(best_path, map_location=device)
        agent.network.load_state_dict(checkpoint['network_state_dict'], strict=False)
        agent.network.eval()
        print(f"Loaded best model from {best_path}")

        # Simple eval loop — no collect_trajectories, reads directly from env/topstep
        full_test_env = SyncVectorEnv([
            lambda: TradingEnv(data=test_chunks[0], data_pool=test_chunks, env_id=0, num_envs=1)
        ])
        full_steps = len(test_chunks[0]['ohlc'])
        obs, _ = full_test_env.reset()
        last_actions = None
        action_counts = np.zeros(3)

        for step in range(full_steps - 1):
            obs_t = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
            with torch.no_grad():
                action_dict = agent.network.get_action(obs_t, prev_actions=last_actions, deterministic=True)
            da = action_dict['direction_action'].cpu().numpy()
            pc = action_dict['pm_confidence'].cpu().numpy()
            mg = action_dict['magnitude'].cpu().numpy()
            last_actions = action_dict['direction_action']
            actions = np.stack([da.astype(np.float32), pc.astype(np.float32),
                                mg.astype(np.float32)], axis=-1)
            obs, rewards, terminated, truncated, infos = full_test_env.step(actions)
            action_counts[int(da[0])] += 1
            # Capture before done resets everything
            e = full_test_env.envs[0]
            ts = e.pm.topstep
            last_payout = ts.payouts
            last_acc_pnl = ts.acc_pnl
            last_pass_rate = ts.exam_success / max(ts.exam_counts, 1)
            last_exam_cost = ts.topstep_cost
            last_exam_counts = ts.exam_counts
            last_exam_success = ts.exam_success
            last_sl = e.stop_loss_count
            last_avg_net, last_total_net = e.get_final_metrics()

        total_acts = action_counts.sum()
        pct_buy = action_counts[0] / total_acts * 100   # 0=buy
        pct_sell = action_counts[1] / total_acts * 100  # 1=sell
        print(f"  Payout:{last_payout:+,.0f} | AccPnl:{last_acc_pnl:+,.0f} | PassRate:{last_pass_rate:.2f} | "
              f"ExamCost:{last_exam_cost:+,.0f} | ExamCnt:{last_exam_counts} | ExamPass:{last_exam_success} | "
              f"AvgNet:{last_avg_net:+,.0f} | TotalNet:{last_total_net:+,.0f} | "
              f"SW:0 SL:{last_sl} | "
              f"Buy:{pct_buy:.1f}% Sell:{pct_sell:.1f}%")
        full_test_env.close()
    else:
        print(f"  No best model found at {best_path}")
    # print(f"{'='*70}")


#steps_per_env: Should be 5-20% of average episode length


    # num_test_envs = min(4, len(test_chunks))
    # test_envs = SyncVectorEnv([
    #     lambda i=i: TradingEnv(data=test_chunks[i % len(test_chunks)], data_pool=test_chunks, env_id=i, num_envs=num_test_envs)
    #     for i in range(num_test_envs)
    # ])

    # test_buffer = MultiActorPPOBuffer(
    #     num_test_envs, MAX_STEPS_PER_EPISODE, agent.obs_shapes, agent.gamma, agent.gae_lambda
    # )

    # # Run test episodes with deterministic actions
    # trained_model.eval()
    # test_pnls = []
    # test_mdds = []
    # test_costs = []
    # test_trades = []
    # test_wins = []
    # test_holds = []

    # current_obs, _ = test_envs.reset()
    # last_actions = None

    # for step in range(MAX_STEPS_PER_EPISODE):
    #     obs_tensor = {
    #         key: torch.as_tensor(arr, dtype=torch.float32, device=device)
    #         for key, arr in current_obs.items()
    #     }
    #     with torch.no_grad():
    #         actions, _, _, _ = trained_model.get_action(obs_tensor, prev_action=last_actions, deterministic=True)

    #     actions_np = actions.cpu().numpy()
    #     next_obs, rewards, terminated, truncated, infos = test_envs.step(actions_np)
    #     dones = np.logical_or(terminated, truncated)
    #     last_actions = actions

    #     for i in range(num_test_envs):
    #         if dones[i]:
    #             if isinstance(infos, dict) and 'final_info' in infos:
    #                 info = infos['final_info'][i] if infos['final_info'][i] is not None else infos
    #             else:
    #                 info = infos
    #             if isinstance(info, dict):
    #                 test_pnls.append(info.get('pfolio_pnl', 0))
    #                 test_mdds.append(info.get('pfolio_mdd', 0))
    #                 test_costs.append(info.get('total_trading_cost', 0))
    #                 test_trades.append(info.get('n_trades', 0))
    #                 test_wins.append(info.get('win_rate', 0))
    #                 test_holds.append(info.get('avg_hold', 0))

    #     current_obs = next_obs

    # test_envs.close()

    # if test_pnls:
    #     print(f"  Episodes completed: {len(test_pnls)}")
    #     print(f"  Avg PnL:      {np.mean(test_pnls):+.1f}")
    #     print(f"  Avg MDD:      {np.mean(test_mdds):.1f}")
    #     print(f"  Avg Cost:     {np.mean(test_costs):.1f}")
    #     print(f"  Avg Trades:   {np.mean(test_trades):.0f}")
    #     print(f"  Avg Win Rate: {np.mean(test_wins)*100:.1f}%")
    #     print(f"  Avg Hold:     {np.mean(test_holds):.0f} steps")
    # else:
    #     print("  No test episodes completed (episodes may be longer than MAX_STEPS_PER_EPISODE)")
    # print(f"{'='*70}")