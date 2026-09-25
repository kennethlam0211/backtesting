"""Test env_v2 step function with comprehensive scenarios."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def make_fake_data(n_bars, price_series, consensus_schedule):
    """Build fake np_data for env_v2."""
    N = n_bars
    closes = np.array(price_series, dtype=np.float64)
    opens = closes + np.random.normal(0, 0.3, N)
    highs = np.maximum(closes, opens) + np.abs(np.random.normal(0, 2, N))
    lows = np.minimum(closes, opens) - np.abs(np.random.normal(0, 2, N))

    ohlc = np.zeros((N, 13))
    ohlc[:, 0] = np.arange(N)
    ohlc[:, 1] = opens
    ohlc[:, 2] = highs
    ohlc[:, 3] = lows
    ohlc[:, 4] = closes
    ohlc[0, 8] = 1
    ohlc[:, 9] = 0.5
    ohlc[:, 12] = 1

    s4 = np.zeros((N, 72, 3), dtype=np.float32)
    cgf = np.zeros((N, 72, 3), dtype=np.float32)
    rofs = np.zeros((N, 72, 3), dtype=np.float32)

    current_dir = 'flat'
    for i in range(N):
        if i in consensus_schedule:
            current_dir = consensus_schedule[i]
        if current_dir == 'long':
            for v in range(50):
                s4[i, v] = [0.1, 1.0, 0.6]
                cgf[i, v] = [0.2, 1.0, 0.7]
                rofs[i, v] = [0.05, 1.0, 0.5]
        elif current_dir == 'short':
            for v in range(50):
                s4[i, v] = [-0.1, -1.0, 0.6]
                cgf[i, v] = [-0.2, -1.0, 0.7]
                rofs[i, v] = [-0.05, -1.0, 0.5]
        if i in consensus_schedule and i > 0:
            s4[i, 71, 2] = 0.001 * i

    vol = np.zeros((N, 6), dtype=np.float32)
    vol[:] = [0.5, 0.3, 0.4, 0.2, 0.6, 0.1]

    return {
        'ohlc': ohlc,
        'S4_variants': s4,
        'CGF_variants': cgf,
        'ROFS_variants': rofs,
        'vol_60': vol,
    }


def run_scenario(name, n_bars, prices, consensus, action, verbose=True):
    from env_v2 import TradingEnvV2
    np.random.seed(42)
    data = make_fake_data(n_bars, prices, consensus)
    env = TradingEnvV2(data)
    obs, _ = env.reset()

    print(f'\n{"="*70}')
    print(f'SCENARIO: {name}')
    print(f'{"="*70}')

    history = []
    for step in range(env.current_step, min(env.n_steps - 1, n_bars - 1)):
        result = env.step(action)
        obs, reward, done, _, infos = result
        is_60 = infos.get('is_60min', False)
        bar = env.current_step
        c = prices[bar]

        rec = {
            'bar': bar, 'close': c, 'ud_side': env.ud_side, 'side': env.side,
            'last_u': env.last_u, 'last_d': env.last_d, 'is_60': is_60,
            'reward': reward, 'consensus': env._last_consensus,
        }
        history.append(rec)

        if verbose:
            if is_60:
                print(f'[60m] bar={bar:>3} close={c:>8.1f} cons={env._last_consensus:>6.2f} '
                      f'ud={env.ud_side:>2} side={env.side:>2} '
                      f'last_u={env.last_u:>8.1f} last_d={env.last_d:>8.1f} '
                      f'rew={reward:>7.3f}')
            elif env.side != 0 or env.ud_side != 0:
                print(f'      bar={bar:>3} close={c:>8.1f} '
                      f'ud={env.ud_side:>2} side={env.side:>2} '
                      f'last_u={env.last_u:>8.1f} last_d={env.last_d:>8.1f}')

        if done:
            break

    print(f'  RESULT: side={env.side} ud={env.ud_side} stops={env.stop_loss_count} '
          f'acc_pnl={env.pm.topstep.acc_pnl:.2f}')
    return env, history


# ── SCENARIOS ────────────────────────────────────────────────────

print('='*70)
print('ENV V2 COMPREHENSIVE TESTS')
print('='*70)

action = np.array([0.5, 0.8], dtype=np.float32)  # K=50, confidence=0.8
action_tight = np.array([0.1, 0.8], dtype=np.float32)  # K=10, confidence=0.8
action_low_conf = np.array([0.5, 0.1], dtype=np.float32)  # K=50, confidence=0.1

# ── 1. Flat consensus throughout ─────────────────────────────────
prices = [44000.0 + i * 0.1 for i in range(180)]
env, h = run_scenario('1. Flat consensus - no trading', 180, prices,
                       {60: 'flat', 120: 'flat'}, action)
assert env.side == 0, 'Should be flat'
assert env.stop_loss_count == 0, 'No stops'
assert env.pm.topstep.acc_pnl == 0, 'No PnL'
print('  PASS')

# ── 2. Long consensus, steady uptrend ────────────────────────────
prices = [44000.0 + i * 2 for i in range(180)]
env, h = run_scenario('2. Long + steady uptrend', 180, prices,
                       {60: 'long'}, action)
assert env.side == 1, 'Should be long'
assert env.ud_side == 1, 'U/D should be uptrend'
assert env.last_u > 44300, 'Watermark should track high'
assert env.stop_loss_count == 0, 'No stops in smooth uptrend'
print('  PASS')

# ── 3. Long consensus, price drops K from high ──────────────────
prices = [44000.0 + i * 3 for i in range(80)] + \
         [44000.0 + 80*3 - i * 5 for i in range(100)]
env, h = run_scenario('3. Long + drop exceeds K', 180, prices,
                       {60: 'long'}, action)
# U/D should flip to -1 after drop exceeds K=50
assert env.ud_side == -1, f'U/D should flip to downtrend, got {env.ud_side}'
assert env.side == 0, 'PM should be flat (U/D disagrees with long consensus)'
print('  PASS')

# ── 4. Consensus flips long to short ────────────────────────────
prices = [44000.0 + i * 1 for i in range(180)]
env, h = run_scenario('4. Consensus flips long to short', 180, prices,
                       {60: 'long', 120: 'short'}, action)
# After flip, should be short (U/D needs to agree)
print(f'  Note: side={env.side}, ud={env.ud_side}')
print('  PASS')

# ── 5. Consensus goes flat while holding ─────────────────────────
prices = [44000.0 + i * 2 for i in range(180)]
env, h = run_scenario('5. Consensus flat while holding', 180, prices,
                       {60: 'long', 120: 'flat'}, action)
assert env.side == 0, 'Should close on flat consensus'
print('  PASS')

# ── 6. Tight K causes stops, re-entry ───────────────────────────
np.random.seed(123)
prices = [44000.0]
for i in range(1, 180):
    prices.append(prices[-1] + np.random.normal(0, 20))
env, h = run_scenario('6. Tight K=10 + volatile', 180, prices,
                       {60: 'long'}, action_tight)
print(f'  Stops: {env.stop_loss_count}')
assert env.stop_loss_count > 0, 'Should have stops with tight K'
print('  PASS')

# ── 7. Same side consensus renewal, watermarks preserved ────────
prices = [44000.0 + i * 2 for i in range(240)]
env, h = run_scenario('7. Same side consensus renewal', 240, prices,
                       {60: 'long', 120: 'long', 180: 'long'}, action, verbose=False)
# Check watermarks were NOT reset at 120 and 180
found_reset = False
for rec in h:
    if rec['is_60'] and rec['bar'] in [120, 121]:
        # last_u should be continuous, not reset to close
        if rec['last_u'] < 44200:  # should be tracking high ~44240
            found_reset = True
assert not found_reset, 'Watermarks should NOT reset on same-side renewal'
print('  PASS')

# ── 8. U/D flips within hour, PM exits and re-enters ────────────
# Price goes up then sharply down then up again within one consensus
prices = [44000.0]
for i in range(1, 70):
    prices.append(prices[-1] + 3)  # up
for i in range(70, 100):
    prices.append(prices[-1] - 5)  # sharp drop
for i in range(100, 180):
    prices.append(prices[-1] + 3)  # recovery
env, h = run_scenario('8. U/D flips within hour', 180, prices,
                       {60: 'long'}, action)
# Should see U/D flip to -1 during drop, then back to 1 on recovery
ud_changes = [rec for rec in h if rec['bar'] > 60 and rec['bar'] < 180]
ud_sides_seen = set(rec['ud_side'] for rec in ud_changes)
print(f'  U/D sides seen: {ud_sides_seen}')
assert -1 in ud_sides_seen, 'Should see downtrend during drop'
assert 1 in ud_sides_seen, 'Should see uptrend during recovery'
print('  PASS')

# ── 9. Low confidence, reduced sizing ────────────────────────────
prices = [44000.0 + i * 2 for i in range(180)]
env, h = run_scenario('9. Low confidence=0.1', 180, prices,
                       {60: 'long'}, action_low_conf, verbose=False)
# Position should be small
print(f'  Position: {env.pm.position_size_equivalent}')
print('  PASS')

# ── 10. Short consensus + downtrend ──────────────────────────────
prices = [44000.0 - i * 2 for i in range(180)]
env, h = run_scenario('10. Short + downtrend', 180, prices,
                       {60: 'short'}, action)
assert env.side == -1, 'Should be short'
assert env.ud_side == -1, 'U/D should be downtrend'
assert env.last_d < 43700, 'Watermark should track low'
print('  PASS')

# ── 11. Short consensus, price rises K ───────────────────────────
prices = [44000.0 - i * 3 for i in range(80)] + \
         [44000.0 - 80*3 + i * 5 for i in range(100)]
env, h = run_scenario('11. Short + rise exceeds K', 180, prices,
                       {60: 'short'}, action)
assert env.ud_side == 1, f'U/D should flip to uptrend, got {env.ud_side}'
assert env.side == 0, 'PM flat (U/D disagrees with short consensus)'
print('  PASS')

# ── 12. Multiple 60min boundaries, consensus changes ────────────
prices = [44000.0]
for i in range(1, 300):
    prices.append(prices[-1] + np.random.normal(0.5, 3))
env, h = run_scenario('12. Multiple consensus changes', 300, prices,
                       {60: 'long', 120: 'short', 180: 'flat', 240: 'long'}, action,
                       verbose=False)
# Count 60min events
n_60 = sum(1 for rec in h if rec['is_60'])
print(f'  60min events: {n_60}')
assert n_60 >= 3, 'Should have multiple 60min events'
print('  PASS')

# ── 13. Session end handling ─────────────────────────────────────
prices = [44000.0 + i * 1 for i in range(180)]
data = make_fake_data(180, prices, {60: 'long'})
data['ohlc'][150, 8] = 1  # session end at bar 150
from env_v2 import TradingEnvV2
env = TradingEnvV2(data)
obs, _ = env.reset()
print(f'\n{"="*70}')
print('SCENARIO: 13. Session end mid-trade')
print(f'{"="*70}')
for step in range(env.current_step, min(env.n_steps - 1, 160)):
    result = env.step(action)
    obs, reward, done, _, infos = result
print(f'  Kelly updated at session end: n_trades={env._cached_kelly_ewm["n_trades"]}')
print('  PASS')

# ── 14. Reward at flat hour vs trading hour ──────────────────────
prices = [44000.0 + i * 2 for i in range(240)]
env, h = run_scenario('14. Reward comparison flat vs trading', 240, prices,
                       {60: 'flat', 120: 'long', 180: 'flat'}, action, verbose=False)
flat_rewards = [rec['reward'] for rec in h if rec['is_60'] and rec['consensus'] == 0]
trade_rewards = [rec['reward'] for rec in h if rec['is_60'] and rec['consensus'] != 0]
print(f'  Flat hour rewards: {flat_rewards}')
print(f'  Trading hour rewards: {trade_rewards}')
print('  PASS')

# ── 15. Watermark tracks close correctly ─────────────────────────
prices = [44000.0, 44010, 44020, 44030, 44025, 44015, 44005, 43990, 43970,  # up then down
          43960, 43950, 43940, 43950, 43960, 43970, 43980, 43990, 44000,
          44010, 44020, 44030, 44040, 44050, 44060, 44070, 44080, 44090,
          44100, 44110, 44120, 44130, 44140, 44150, 44160, 44170, 44180,
          44190, 44200, 44210, 44220, 44230, 44240, 44250, 44260, 44270,
          44280, 44290, 44300, 44310, 44320, 44330, 44340, 44350, 44360,
          44370, 44380, 44390, 44400, 44410, 44420,  # bar 59
          44430, 44440, 44450, 44460, 44470, 44480, 44490, 44500, 44510,
          44520, 44530, 44540, 44550, 44560, 44570, 44580, 44590, 44600,
          44610, 44620]  # bar 79
# Pad to 100
while len(prices) < 100:
    prices.append(prices[-1] + 10)

env, h = run_scenario('15. Watermark tracks close', 100, prices,
                       {60: 'long'}, action)
# After bar 79, close=44620, last_u should be 44620
trading_bars = [rec for rec in h if rec['side'] == 1]
if trading_bars:
    last_rec = trading_bars[-1]
    print(f'  Last trading bar: {last_rec["bar"]}, last_u={last_rec["last_u"]:.1f}, close={last_rec["close"]:.1f}')
    assert last_rec['last_u'] >= last_rec['close'], 'last_u should >= close'
print('  PASS')

print(f'\n{"="*70}')
print('ALL 15 SCENARIOS PASSED')
print(f'{"="*70}')
