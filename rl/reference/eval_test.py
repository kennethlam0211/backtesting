"""
Evaluate saved best model on the held-out test chunk.
Uses EXACT same data pipeline as ppo.py for consistent results.

Usage:
    python eval_test.py [model_dir]
    Default: auto-finds latest models/ppo_* folder
    Example: python eval_test.py models/ppo_20260320_233644
"""
import os, sys, glob, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_pipeline'))

from gym.vector import SyncVectorEnv
from env import TradingEnv
from network import TradingNetworkLite
from ppo import create_chunks
from post_preprocessing import convert_to_numpy
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Same data pipeline as ppo.py ----
STEPS_PER_EPISODE = 30 * 1020
MIN_STEPS_PER_EPISODE = 25 * 1020
MAX_STEPS_PER_EPISODE = 32 * 1020

data = pd.read_pickle('./data/normalised_trading_data.pkl')
print(f"Loaded data: {len(data):,} rows")

all_chunks_df = create_chunks(data,
                               steps_per_episode=STEPS_PER_EPISODE,
                               minimum_steps_per_episode=MIN_STEPS_PER_EPISODE,
                               max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                               is_daily=False)
all_chunks = [convert_to_numpy(chunk.reset_index(drop=True)) for chunk in all_chunks_df]

# Same split as ppo.py
test_chunks = all_chunks[-1:]
print(f"Total chunks: {len(all_chunks)}, Test chunk: {len(test_chunks[0]['ohlc'])} steps")

# ---- Find model ----
if len(sys.argv) > 1:
    model_dir = sys.argv[1]
else:
    dirs = sorted(glob.glob('models/ppo_*'))
    model_dir = dirs[-1] if dirs else 'models'

best_path = f'{model_dir}/best_trading_model.pth'
if not os.path.exists(best_path):
    print(f"Model not found: {best_path}")
    sys.exit(1)
print(f"Loading model from {best_path}")

# ---- Load model ----
network = TradingNetworkLite(num_actions=2, l1_coef=1e-5, use_input_gates=True).to(device)
checkpoint = torch.load(best_path, map_location=device)
missing, unexpected = network.load_state_dict(checkpoint['network_state_dict'], strict=False)
if missing:
    print(f"Missing keys: {missing}")
network.eval()
print(f"Model loaded (update {checkpoint.get('update', '?')}), {sum(p.numel() for p in network.parameters()):,} params")

# ---- Run test ----
test_env = SyncVectorEnv([
    lambda: TradingEnv(data=test_chunks[0], data_pool=test_chunks, env_id=0, num_envs=1)
])
full_steps = len(test_chunks[0]['ohlc'])
print(f"Running {full_steps} steps...")

obs, _ = test_env.reset()
last_actions = None
action_counts = np.zeros(3)  # clear, buy, sell

for step in range(full_steps - 1):
    obs_t = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
    with torch.no_grad():
        action_dict = network.get_action(obs_t, prev_actions=last_actions, deterministic=True)

    da = action_dict['direction_action'].cpu().numpy()
    pc = action_dict['pm_confidence'].cpu().numpy()
    sr = action_dict['stop_ratio'].cpu().numpy()
    mg = action_dict['magnitude'].cpu().numpy()
    last_actions = action_dict['direction_action']

    actions = np.stack([
        da.astype(np.float32),
        pc.astype(np.float32),
        sr.astype(np.float32),
        mg.astype(np.float32),
    ], axis=-1)

    obs, rewards, terminated, truncated, infos = test_env.step(actions)
    action_counts[int(da[0])] += 1

    # Capture results BEFORE done resets everything
    e = test_env.envs[0]
    ts = e.pm.topstep
    last_payout = ts.payouts
    last_acc_pnl = ts.acc_pnl
    last_pass_rate = ts.exam_success / max(ts.exam_counts, 1)
    last_exam_cost = ts.topstep_cost
    last_exam_counts = ts.exam_counts
    last_exam_success = ts.exam_success
    last_sw = e.stop_win_count
    last_sl = e.stop_loss_count
    last_avg_net, last_total_net = e.get_final_metrics()

    if step % 5000 == 0 and step > 0:
        print(f"  step {step:>6,}/{full_steps} | AccPnl:{last_acc_pnl:+,.0f} | Payout:{last_payout:+,.0f} | "
              f"ExamCnt:{last_exam_counts} | SW:{last_sw} SL:{last_sl}")

# ---- Results (from last captured state, before done reset) ----
total_acts = action_counts.sum()
pct_clear = action_counts[0] / total_acts * 100
pct_buy = action_counts[1] / total_acts * 100
pct_sell = action_counts[2] / total_acts * 100

composite = last_acc_pnl * 0.45 + last_payout + last_exam_cost

print(f"\n{'='*70}")
print(f"TEST SET RESULTS ({full_steps} steps)")
print(f"{'='*70}")
print(f"  Payout:    {last_payout:+,.0f}")
print(f"  AccPnl:    {last_acc_pnl:+,.0f}")
print(f"  PassRate:  {last_pass_rate:.2f}")
print(f"  ExamCost:  {last_exam_cost:+,.0f}")
print(f"  ExamCnt:   {last_exam_counts}  (passed: {last_exam_success})")
print(f"  AvgNet:    {last_avg_net:+,.0f}")
print(f"  TotalNet:  {last_total_net:+,.0f}")
print(f"  SW:{last_sw}  SL:{last_sl}")
print(f"  Actions:   Clear:{pct_clear:.1f}%  Buy:{pct_buy:.1f}%  Sell:{pct_sell:.1f}%")
print(f"  Composite: {composite:+,.0f}  (AccPnl*0.45 + Payout + ExamCost)")
print(f"{'='*70}")

test_env.close()
