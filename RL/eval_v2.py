"""Evaluate best model on test chunks.

Usage:
    python eval_v2.py [model_dir]
    python eval_v2.py models/ppo_v2_20260403_111349
    python eval_v2.py  # uses latest model dir
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import params
# Use eval topstep (no reset limits, different params)
import topstep_for_evaluate
import sys
sys.modules['topstep'] = topstep_for_evaluate

from env_v2 import TradingEnvV2
from network_v2 import TradingNetworkV2
from data_pipeline.to_numpy import convert_to_numpy
from run_ppo_v2 import create_chunks, CONFIG
import pandas as pd



def evaluate(model_dir, test_chunks, device='cpu'):
    """Run best model on test chunks continuously (kelly carries over)."""
    best_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(best_path):
        print(f'No best model at {best_path}')
        return

    net = TradingNetworkV2(selector_mode=params.SELECTOR_MODE)
    net.load_state_dict(torch.load(best_path, map_location=device))
    net.to(device)
    net.eval()
    print(f'Loaded: {best_path}')
    print(f'Network: {net.get_param_count():,} params')
    print(f'Test chunks: {len(test_chunks)}')

    # Run each chunk separately (fresh kelly each)
    print(f'\n{"="*70}')
    print(f'PER-CHUNK EVALUATION')
    print(f'{"="*70}')

    for ci, chunk in enumerate(test_chunks):
        env = TradingEnvV2(data=chunk, data_pool=[chunk], env_id=0, num_envs=1)
        obs, _ = env.reset()
        done = False
        step_count = 0

        while not done:
            obs_t = {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0) for k, v in obs.items()}
            sel_keys = ['session_summary', 'pfolio_info']
            sel_keys += [f'{fam}_variants' for fam in params.STRAT_FAMILIES]
            if params.SELECTOR_MODE == 'rl':
                sel_keys += [f'{fam}_metrics_ts' for fam in params.STRAT_FAMILIES]
            sel_obs = {k: obs_t[k] for k in sel_keys if k in obs_t}
            k_obs = {k: obs_t[k] for k in ['vol_60']}
            with torch.no_grad():
                action_out = net.get_action(sel_obs, k_obs, deterministic=True)

            for fam in params.STRAT_FAMILIES:
                env.family_attns[fam] = action_out[f'{fam.lower()}_attn'].cpu().numpy()[0]

            action_list = [action_out['k_index'].item()]
            action_list += [action_out[f'conf_{fam.lower()}'].item() for fam in params.STRAT_FAMILIES]
            action = np.array(action_list, dtype=np.float32)
            obs, reward, done, _, info = env.step(action)
            step_count += 1

        ts = env.pm.topstep
        th = ts.payouts + ts.topstep_cost
        trades = len(env.pm.sim_net_pnl_per_con_list)
        print(f'  Chunk {ci+1}: TH:{th:>+8,.0f} | Pay:{ts.payouts:>+8,.0f} Acc:{ts.acc_pnl:>+7,.0f} | '
              f'Cost:{ts.topstep_cost:>7,.0f} LWM:{ts.take_home_lwm:>+7,.0f} | '
              f'PR:{ts.exam_success/max(ts.exam_counts,1):.2f} Ex:{ts.exam_counts:>3} | '
              f'SL:{env.stop_loss_count:>4} Tr:{trades:>4} Steps:{step_count:>4} | '
              f'K:{env.current_k:.0f} Cons:{env._last_consensus:+.3f}')

    # Run all chunks continuously (kelly carries over)
    print(f'\n{"="*70}')
    print(f'CONTINUOUS EVALUATION (kelly carries over)')
    print(f'{"="*70}')

    env = TradingEnvV2(data=test_chunks[0], data_pool=test_chunks, env_id=0, num_envs=1)

    for ci in range(len(test_chunks)):
        obs, _ = env.reset()
        done = False
        step_count = 0

        while not done:
            obs_t = {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0) for k, v in obs.items()}
            sel_keys = ['session_summary', 'pfolio_info']
            sel_keys += [f'{fam}_variants' for fam in params.STRAT_FAMILIES]
            if params.SELECTOR_MODE == 'rl':
                sel_keys += [f'{fam}_metrics_ts' for fam in params.STRAT_FAMILIES]
            sel_obs = {k: obs_t[k] for k in sel_keys if k in obs_t}
            k_obs = {k: obs_t[k] for k in ['vol_60']}
            with torch.no_grad():
                action_out = net.get_action(sel_obs, k_obs, deterministic=True)

            for fam in params.STRAT_FAMILIES:
                env.family_attns[fam] = action_out[f'{fam.lower()}_attn'].cpu().numpy()[0]

            action_list = [action_out['k_index'].item()]
            action_list += [action_out[f'conf_{fam.lower()}'].item() for fam in params.STRAT_FAMILIES]
            action = np.array(action_list, dtype=np.float32)
            obs, reward, done, _, info = env.step(action)
            step_count += 1

        ts = env.pm.topstep
        th = ts.payouts + ts.topstep_cost
        trades = len(env.pm.sim_net_pnl_per_con_list)
        kelly = env._cached_kelly_ewm
        print(f'  Chunk {ci+1}: TH:{th:>+8,.0f} | Pay:{ts.payouts:>+8,.0f} Acc:{ts.acc_pnl:>+7,.0f} | '
              f'Cost:{ts.topstep_cost:>7,.0f} LWM:{ts.take_home_lwm:>+7,.0f} | '
              f'PR:{ts.exam_success/max(ts.exam_counts,1):.2f} Ex:{ts.exam_counts:>3} | '
              f'SL:{env.stop_loss_count:>4} Tr:{trades:>4} Steps:{step_count:>4} | '
              f'Kelly:{kelly["confidence"]:.3f} WR:{kelly["win_rate"]:.2f}')
    print(f'{"="*70}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find model dir
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs = sorted(glob.glob(os.path.join(base, 'models/ppo_v2_*')))
        model_dir = dirs[-1] if dirs else ''
        print(f'Using latest: {model_dir}')

    # Load data and create test chunks
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'normalised_trading_data.pkl')
    print(f'Loading: {data_path}')
    df = pd.read_pickle(data_path)
    print(f'Shape: {df.shape}')

    all_chunks_df = create_chunks(df, CONFIG['steps_per_episode'], CONFIG['min_steps_per_episode'], CONFIG['max_steps_per_episode'])

    print(f'Total chunks: {len(all_chunks_df)}')

    # Val chunks
    print(f'Converting val chunks...')
    val_chunks_df = all_chunks_df[-6:-3]
    val_chunks = [convert_to_numpy(c) for c in val_chunks_df]
    print(f'\n*** VAL EVALUATION ***')
    evaluate(model_dir, val_chunks, device)

    # Test chunks
    print(f'\nConverting test chunks...')
    test_chunks_df = all_chunks_df[-3:]
    test_chunks = [convert_to_numpy(c) for c in test_chunks_df]
    print(f'\n*** TEST EVALUATION ***')
    evaluate(model_dir, test_chunks, device)
