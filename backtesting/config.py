"""
The backtest run: which strategies, which sessions, the TP x SL grid and the costs. Edit here, then

    python -m backtesting.backtest

Paths (training data, tick.dat, results/) and feature settings stay in params/params.py.
Prices and distances are in units of price x4: 1 unit = 0.25 index point.
"""
from backtesting.strategy import rsi
from params import VOL_METHODS, WINDOW_SIZE

# Bar size traded, any of params.FREQS (the training data is the playground): signals at its bars' close, entry at the
# next bar, exits walked bar by bar. Above the base bar (the smallest), the bars come from their bar file with the
# training columns as of each bar's close; flat at the start of the bar holding FLAT_AT (for '5': the 22:55 bar)
FREQ = '1'

# Strategies: factories from backtesting/strategy.py with their settings; each runs over the whole TP x SL grid.
# RSI mean reversion on FREQ bars, each side on its own (a long depends only on its level, a short only on its own):
# long when the RSI first goes below a level of RSI_LONG_GRID, short when it first goes above one of RSI_SHORT_GRID
RSI_LONG_GRID = [5, 10, 15, 20, 25, 30]
RSI_SHORT_GRID = [95, 90, 85, 80, 75, 70]
STRATEGIES = [rsi(low=level, freq=FREQ) for level in RSI_LONG_GRID] + [rsi(high=level, freq=FREQ) for level in RSI_SHORT_GRID]

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
SPLIT = '2026-01-01'
# Training-data columns kept on each trade from its signal bar (known when the order goes in); the analysis labels
# them in n quantile buckets over all bars
# The FREQ bars' volatility (params.VOL_METHODS over params.WINDOW_SIZE bars, e.g. 20_std_1): 1 = calmest 20% .. 5
LABEL_FEATURES = {f'{WINDOW_SIZE}_{VOL_METHODS}_{FREQ}': 5}
# The final run, reported with its statistics and charts in results/final/: (strategy name, tp, sl); None = best net
FINAL = None  # e.g. ('rsi_15_85', 40, 24)
