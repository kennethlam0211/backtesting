"""Run PPO v2 training with chunked data, train/val/test split.

Chunks data by trading days, splits into train/val/test,
creates vectorized envs, runs PPO v2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from gym.vector import SyncVectorEnv
import pandas as pd
import datetime
from env import TradingEnv
from ppo import PPOTrainer
from data_pipeline.to_numpy import convert_to_numpy
import params
from utils.app_logger import get_logger
logger = get_logger('run_ppo', level=params.LOG_LEVEL)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


# ── Chunking ────────────────────────────────────────────────────

def create_date_chunk_info(df):
    """Group bars by trading date, get first/last index per date."""
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = (df['ts'] - datetime.timedelta(hours=5)).dt.date

    chunk_info = df.groupby('date').agg(
        count=('index_1', 'count'),
        first_index=('index_1', 'first'),
        last_index=('index_1', 'last'),
    ).reset_index()

    chunk_info = chunk_info[chunk_info['count'] > 500]
    chunk_info.reset_index(drop=True, inplace=True)
    return chunk_info


def create_episodes(chunk_info, steps_per_episode):
    """Group days into episodes of ~steps_per_episode bars."""
    episodes = []
    current_episode = []
    current_count = 0

    for _, row in chunk_info.iterrows():
        current_episode.append(row)
        current_count += row['count']

        if current_count >= steps_per_episode:
            episodes.append(current_episode)
            current_episode = []
            current_count = 0

    if current_episode:
        episodes.append(current_episode)

    return episodes


def create_chunks(df, steps_per_episode, min_steps, max_steps=None):
    """Create chunks from DataFrame, split by trading days."""
    chunk_info = create_date_chunk_info(df)

    WINDOW = 60
    episodes = create_episodes(chunk_info, steps_per_episode)
    ranges = [(max(int(ep[0]['first_index']) - WINDOW, 0), int(ep[-1]['last_index']))
              for ep in episodes if len(ep) > 0]

    chunks = []
    for start, end in ranges:
        chunk = df.loc[start:end].copy()
        if len(chunk) >= min_steps:
            if max_steps:
                chunk = chunk.iloc[:max_steps]
            chunks.append(chunk.reset_index(drop=True))

    return chunks


# ── Config ──────────────────────────────────────────────────────

CONFIG = {
    # Data chunking (1min bars: ~1020 bars/trading day)
    'steps_per_episode': 60 * 1260,     # 60 trading days
    'min_steps_per_episode': 50 * 1260,
    'max_steps_per_episode': 65 * 1260,

    # Envs
    'num_train_envs': 12,
    'num_val_envs': 3,
    'num_test_envs': 3,

    # PPO
    'steps_per_env': 256,              # 60min decision points per rollout (~16 trading days)
    'total_updates': 500,
    'lr': 2e-4,
    'gamma_selector': 0.99,
    'gamma_k': 0.97,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.1,
    'value_coef': 0.5,
    'entropy_coef': 0.03,
    'l1_coef': 5e-5,
    'max_grad_norm': 0.5,
    'n_epochs': 5,
    'batch_size': 512,
    'target_kl': 0.02,

    # Validation & checkpointing
    'validation_freq': 5,
    'val_steps_per_env': 320,          # ~20 days at 60min (stateful, 3 checks = full chunk)
    'checkpoint_freq': 50,
    'early_stop_patience': 30,            # 30 val checks without improvement

    # Regularization
    'weight_decay': 5e-4,
}

# Keys that go to PPOTrainer (exclude data-only keys)
PPO_KEYS = [
    'num_train_envs', 'num_val_envs', 'num_test_envs',
    'steps_per_env', 'total_updates', 'lr', 'gamma_selector', 'gamma_k',
    'gae_lambda', 'clip_epsilon', 'value_coef', 'entropy_coef',
    'l1_coef', 'max_grad_norm', 'n_epochs', 'batch_size', 'target_kl',
    'validation_freq', 'val_steps_per_env', 'checkpoint_freq', 'early_stop_patience', 'weight_decay',
]


if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')

    # ── Load data ────────────────────────────────────────────
    pkl_path = os.path.join(DATA_DIR, 'normalised_trading_data.pkl')
    logger.info(f'Loading: {pkl_path}')
    df = pd.read_pickle(pkl_path)
    logger.info(f'Shape: {df.shape}')

    # Filter post-load to start from 2019-01-01 (skip 2018 burnout/regime data)
    import pandas as _pd
    cutoff = _pd.Timestamp('2019-01-01')
    df['ts'] = _pd.to_datetime(df['ts'])
    before_n = len(df)
    df = df[df['ts'] >= cutoff].reset_index(drop=True)
    logger.info(f'Filtered ts >= {cutoff.date()}: {before_n} → {len(df)} rows ({before_n - len(df)} dropped)')

    # ── Create chunks ────────────────────────────────────────
    logger.info('\nCreating chunks...')
    all_chunks_df = create_chunks(
        df,
        steps_per_episode=CONFIG['steps_per_episode'],
        min_steps=CONFIG['min_steps_per_episode'],
        max_steps=CONFIG['max_steps_per_episode'],
    )
    logger.info(f'Created {len(all_chunks_df)} chunks')

    # Convert to numpy
    logger.info(f'Converting {len(all_chunks_df)} chunks to numpy...')
    all_chunks = []
    for ci, chunk in enumerate(all_chunks_df):
        all_chunks.append(convert_to_numpy(chunk))
        if (ci + 1) % 100 == 0 or ci == len(all_chunks_df) - 1:
            logger.info(f'  {ci + 1}/{len(all_chunks_df)} chunks converted')

    # Split: last 3 = test, 3 before = val, rest = train
    test_chunks = all_chunks[-3:]
    val_chunks = all_chunks[-6:-3]
    train_chunks = all_chunks[:-6]

    logger.info(f'Train: {len(train_chunks)} chunks')
    logger.info(f'Val:   {len(val_chunks)} chunks')
    logger.info(f'Test:  {len(test_chunks)} chunks')
    total_train_bars = sum(len(c['ohlc']) for c in train_chunks)
    total_train_steps = total_train_bars // 60  # ~60 bars per 60min decision
    logger.info(f'Train data: {total_train_bars:,} bars ({total_train_steps:,} steps)')

    # ── Env factories ────────────────────────────────────────
    mode = params.TRAINING_MODE

    # Exam stats for activation mode (from best exam run)
    exam_kwargs = {}
    if mode == 'train_activation':
        exam_kwargs = {
            'exam_success': params.ACTIVATION_EXAM_SUCCESS,
            'exam_counts': params.ACTIVATION_EXAM_COUNTS,
        }

    def make_train_env(i):
        return TradingEnv(
            data=train_chunks[i % len(train_chunks)],
            data_pool=train_chunks,
            env_id=i, num_envs=CONFIG['num_train_envs'],
            obs_noise_std=0.02, mode=mode, **exam_kwargs)

    def make_val_env(i):
        return TradingEnv(
            data=val_chunks[i % len(val_chunks)],
            data_pool=val_chunks,
            env_id=i, num_envs=CONFIG['num_val_envs'],
            mode=mode, **exam_kwargs)

    def make_test_env(i):
        return TradingEnv(
            data=test_chunks[i % len(test_chunks)],
            data_pool=test_chunks,
            env_id=i, num_envs=CONFIG['num_test_envs'])

    # ── Print config ─────────────────────────────────────────
    logger.info(f'\nConfig:')
    logger.info('=' * 50)
    for k, v in CONFIG.items():
        logger.info(f'  {k:25}: {v}')
    logger.info('=' * 50)

    # ── Train ────────────────────────────────────────────────
    trainer = PPOTrainer(
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        make_test_env=make_test_env,
        device=device,
        total_train_steps=total_train_steps,
        **{k: CONFIG[k] for k in PPO_KEYS},
    )

    # ── Bootstrap from pretrained model if configured ──
    pretrained_path = getattr(params, 'PRETRAINED_MODEL_PATH', '')
    if pretrained_path:
        abs_path = pretrained_path if os.path.isabs(pretrained_path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_path)
        if os.path.exists(abs_path):
            import torch
            ckpt = torch.load(abs_path, map_location=device)
            state = ckpt['network'] if isinstance(ckpt, dict) and 'network' in ckpt else ckpt
            trainer.network.load_state_dict(state, strict=False)
            logger.info(f'Loaded pretrained model: {abs_path}')
        else:
            logger.warning(f'PRETRAINED_MODEL_PATH not found: {abs_path}')

    trainer.train()

    # ── Test evaluation (skip for train_exam/train_activation) ──
    if mode not in ('train_exam', 'train_activation'):
        import torch
        best_path = f'{trainer.model_dir}/best_model.pth'
        if os.path.exists(best_path):
            trainer.network.load_state_dict(torch.load(best_path, map_location=device))
            trainer.network.eval()
            logger.info(f'\n{"="*70}')
            logger.info(f'TEST EVALUATION (best model on {len(test_chunks)} held-out chunks, combined)')
            logger.info(f'{"="*70}')

            combined = {}
            for key in test_chunks[0]:
                combined[key] = np.concatenate([c[key] for c in test_chunks], axis=0)

            test_env = SyncVectorEnv([
                lambda: TradingEnv(data=combined, data_pool=[combined], env_id=0, num_envs=1, mode='eval')
            ])
            obs, _ = test_env.reset()
            e = test_env.envs[0]
            done = False
            while not done:
                obs_t = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
                _test_sel_keys = ['session_summary', 'pfolio_info']
                _test_sel_keys += [f'{fam}_variants' for fam in params.STRAT_FAMILIES]
                if params.SELECTOR_MODE == 'rl':
                    _test_sel_keys += [f'{fam}_metrics_ts' for fam in params.STRAT_FAMILIES]
                sel_obs = {k: obs_t[k] for k in _test_sel_keys if k in obs_t}
                k_obs = {k: obs_t[k] for k in ['vol_60']}
                with torch.no_grad():
                    action_out = trainer.network.get_action(sel_obs, k_obs, deterministic=True)
                k_idx = action_out['k_index'].cpu().numpy()
                # Per-family confidences and attn — loop over STRAT_FAMILIES
                confs = [action_out[f'conf_{fam.lower()}'].cpu().numpy() for fam in params.STRAT_FAMILIES]
                e = test_env.envs[0]
                for fam in params.STRAT_FAMILIES:
                    e.family_attns[fam] = action_out[f'{fam.lower()}_attn'].cpu().numpy()[0]
                actions = np.stack([k_idx] + confs, axis=-1).astype(np.float32)
                obs, rewards, terminated, truncated, infos = test_env.step(actions)
                done = terminated[0] or truncated[0]

            ts = e.pm.topstep
            th = ts.payouts + ts.topstep_cost
            avg_days = np.mean(ts.days_taken_to_pass_exams) if ts.days_taken_to_pass_exams else 0
            logger.info(f'  Combined: TH:{th:>+8,.0f} Pay:{ts.payouts:>+8,.0f} Cost:{ts.topstep_cost:>7,.0f} | '
                  f'ExPR:{ts.exam_success/max(ts.exam_counts,1):.2f} Ex:{ts.exam_success}/{ts.exam_counts} AvgD:{avg_days:.1f} | '
                  f'ActPR:{ts.payout_cnt/max(ts.activation_cnt,1):.2f} Pay:{ts.payout_cnt}/{ts.activation_cnt} BtF:{ts.back_to_fund_cnt} | '
                  f'SL:{e.stop_loss_count:>4} Cm:{ts.total_commission:>+7,.0f}')
            test_env.close()
            logger.info(f'{"="*70}')
        else:
            logger.info(f'No best model found at {best_path}')
