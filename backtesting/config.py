"""
The backtest run: which signals, the ensemble, the stops, which sessions and the costs. Edit here, then

    python -m backtesting.backtest

Paths (training data, tick.dat, results/) and feature settings stay in params/params.py.
Prices and distances are in units of price x4: 1 unit = 0.25 index point.
"""
from backtesting.ensemble import ensemble_function
from backtesting.stops import grid
from backtesting.strategy import rsi_long, rsi_short
from params import VOL_METHODS, WINDOW_SIZE

# Bar size traded, any of params.FREQS (the training data is the playground): signals at its bars' close, entry at the
# next bar, exits walked bar by bar. Above the base bar (the smallest), the bars come from their bar file with the
# training columns as of each bar's close; flat at the start of the bar holding FLAT_AT (for '5': the 22:55 bar)
FREQ = '1'

# Signals: a factory from backtesting/strategy.py, its bar sizes (FREQ or larger) and a list per param; every
# combination is one signal column (rsi_long_5_10: 1 while the RSI of 5-min bars is below 10)
SIGNALS = {
    rsi_long: {'freq': ['1', '5', '15'], 'params': {'level': [5, 10, 15, 20, 25, 30]}},
    rsi_short: {'freq': ['1', '5', '15'], 'params': {'level': [95, 90, 85, 80, 75, 70]}},
    # ud: {'freq': ['1', '5', '15'], 'params': {}},  # later, once its rule is settled
}
# Ensemble: a function (backtesting/ensemble.py) turning the signal columns into one `side`, run as one strategy; a
# trade enters on the bar it turns 1 / -1 only (one signal alone: list just that one in SIGNALS)
ENSEMBLE = ensemble_function

# Stops: a function (backtesting/stops.py) giving each signal its tp / sl (units); grid = every pair
STOP = grid(tp=[4, 5, 6, 7, 8, 16, 24, 40], sl=[4, 5, 6, 7, 8, 16, 24, 40])

# Sessions: 'YYYY-MM-DD', inclusive; None = from the first (after the burn-in) / to the last
START = None
END = None

# Costs and the session cut-off
POINT_VALUE = 12.5  # $ per price unit (ES)
COMMISSION = 1.9    # $ per side (entry and exit), so 3.8 a trade
SLIPPAGE = 1        # units per market fill: the entry, a stop-loss, an opposite-signal or FLAT_AT exit; none on a take-profit (limit)
FLAT_AT = '22:58'   # shifted clock: every position is closed when this bar starts, before the next date

# Label analysis (python -m backtesting.analysis): trades before SPLIT pick a slice, trades from it test it
SPLIT = '2026-01-01'
# Training-data columns kept on each trade from its signal bar (known when the order goes in); the analysis labels
# them in n quantile buckets over all bars
# The FREQ bars' volatility (params.VOL_METHODS over params.WINDOW_SIZE bars, e.g. 20_std_1): 1 = calmest 20% .. 5
LABEL_FEATURES = {f'{WINDOW_SIZE}_{VOL_METHODS}_{FREQ}': 5}
# The final run, reported with its statistics and charts in results/final/: (strategy name, stops); None = best net
FINAL = None  # e.g. ('ensemble_function', '40/24')
