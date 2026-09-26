"""Quick integration test: run PPO v2 with fake data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env import TradingEnv
from ppo import PPOTrainer


def make_fake_data(n_bars=2000):
    """Build fake data large enough for PPO rollouts."""
    np.random.seed(42)

    # Price: random walk with drift
    prices = [44000.0]
    for i in range(1, n_bars):
        prices.append(prices[-1] + np.random.normal(0.5, 10))
    closes = np.array(prices)
    opens = closes + np.random.normal(0, 1, n_bars)
    highs = np.maximum(closes, opens) + np.abs(np.random.normal(0, 3, n_bars))
    lows = np.minimum(closes, opens) - np.abs(np.random.normal(0, 3, n_bars))

    ohlc = np.zeros((n_bars, 13))
    ohlc[:, 0] = np.arange(n_bars)
    ohlc[:, 1] = opens
    ohlc[:, 2] = highs
    ohlc[:, 3] = lows
    ohlc[:, 4] = closes
    ohlc[0, 8] = 1  # session_end for begin_step
    ohlc[:, 9] = 0.5
    ohlc[:, 12] = 1

    # Strategy variants: long signal with some sharpe
    s4 = np.zeros((n_bars, 72, 3), dtype=np.float32)
    cgf = np.zeros((n_bars, 72, 3), dtype=np.float32)
    rofs = np.zeros((n_bars, 72, 3), dtype=np.float32)

    for i in range(60, n_bars):
        # S4: mostly long
        for v in range(40):
            s4[i, v] = [0.1, 1.0, 0.5 + np.random.normal(0, 0.1)]
        for v in range(40, 72):
            s4[i, v] = [-0.05, -1.0, 0.2]
        # CGF: strong long
        for v in range(50):
            cgf[i, v] = [0.2, 1.0, 0.7 + np.random.normal(0, 0.1)]
        # ROFS: mixed
        for v in range(30):
            rofs[i, v] = [0.05, 1.0, 0.3]
        for v in range(30, 50):
            rofs[i, v] = [-0.1, -1.0, 0.2]

    # Make 60min boundaries detectable
    for b in range(60, n_bars, 60):
        if b < n_bars:
            s4[b, 71, 2] = 0.001 * b

    vol = np.zeros((n_bars, 6), dtype=np.float32)
    vol[:] = [0.5, 0.3, 0.4, 0.2, 0.6, 0.1]

    return {
        'ohlc': ohlc,
        'S4_variants': s4,
        'CGF_variants': cgf,
        'ROFS_variants': rofs,
        'vol_60': vol,
    }


if __name__ == '__main__':
    print('Building fake data...')
    data = make_fake_data(5000)

    print('Creating env...')
    env = TradingEnv(data)

    print('Creating PPO trainer...')
    trainer = PPOTrainer(env, config={
        'rollout_60min_steps': 16,
        'total_updates': 5,
        'n_epochs': 3,
        'batch_size': 8,
        'lr': 1e-3,
    })

    print('\nRunning training...')
    trainer.train()

    print('\nDone.')
