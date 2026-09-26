"""
The backtest run: which strategies, which sessions, the TP x SL grid and the costs. Edit here, then

    python -m backtesting.backtest

Paths (training data, tick.dat, results/) and feature settings stay in params/params.py.
Prices and distances are in units of price x4: 1 unit = 0.25 index point.
"""
from backtesting.strategy import rsi

# Strategies: factories from backtesting/strategy.py with their settings; each runs over the whole TP x SL grid
# 1-min RSI mean reversion (long below low, short above high) at 5/95, 10/90, ... 30/70
STRATEGIES = [rsi(low=low, high=100 - low) for low in (5, 10, 15, 20, 25, 30)]

# Sessions: 'YYYY-MM-DD', inclusive; None = from the first (after the burn-in) / to the last
START = None
END = None

# Every TP x SL pair is run, units
TP_GRID = [4, 5, 6, 7, 8, 16, 24, 40]
SL_GRID = [4, 5, 6, 7, 8, 16, 24, 40]

# Costs and the session cut-off
POINT_VALUE = 12.5  # $ per price unit (ES)
COMMISSION = 1.9    # $ per side (entry and exit), so 3.8 a trade
SLIPPAGE = 1        # units per market fill: the entry, a stop-loss, the 22:59 exit; none on a take-profit (limit)
FLAT_AT = '22:59'   # shifted clock: every position is closed when this bar starts, before the next date

# Label analysis (python -m backtesting.analysis): trades before SPLIT pick a slice, trades from it test it
SPLIT = '2024-01-01'
# Training-data columns joined onto the trades at the signal bar (`ts`), as labels in n quantile buckets over all bars
LABEL_FEATURES = {'20_std_1': 5}  # 1-min 20-bar std: 1 = calmest 20% of bars .. 5 = most volatile
# The final run, reported with its statistics and charts in results/final/: (strategy name, tp, sl); None = best net
FINAL = None  # e.g. ('rsi_15_85', 40, 24)
