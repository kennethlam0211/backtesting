"""
The backtest run: which signals, the ensemble, the stops, which sessions and the costs. Edit here, then

    python -m backtesting.backtest

Paths (training data, tick.dat, results/) and feature settings stay in params/params.py.
Prices and distances are in units of price x4: 1 unit = 0.25 index point.
"""
from backtesting.ensemble import ensemble_function
from backtesting.stops import grid, vol_grid  # noqa: F401
from backtesting.strategy import rsi_long, rsi_short
from params import VOL_METHODS, WINDOW_SIZE

# Bar size traded, any of params.FREQS but 'session' (no session bars in tick.dat; the training data is the
# playground): signals at its bars' close, entry at the next bar, exits walked bar by bar. Above the base bar (the smallest), the bars come from their bar file with the
# training columns as of each bar's close; flat at the start of the bar holding FLAT_AT (for '5': the 22:55 bar)
FREQ = '15'
# The FREQ bars' volatility, units: params.VOL_METHODS over params.WINDOW_SIZE bars (e.g. 20_std_5)
VOL_COLUMN = f'{WINDOW_SIZE}_{VOL_METHODS}_{FREQ}'

# Signals: a factory from backtesting/strategy.py, its bar sizes (FREQ or larger) and a list per param (or a list of
# such grids); every combination is one signal column (rsi_long_5_10: 1 while the RSI of 5-min bars is below 10).
# reverse: the other side (rsi_long_5_10_reversed: -1 while below 10, momentum); [False, True] runs both
SIGNALS = {
    rsi_long: {'freq': ['5', '15', '60'], 'params': {'level': [1,2,3,5, 10, 15], 'reverse': [True,False]}},
    rsi_short: {'freq': ['5', '15', '60'], 'params': {'level': [99,98,97,95, 90, 85], 'reverse': [True,False]}},
    # ud: {'freq': ['1', '5', '15'], 'params': {}},  # later, once its rule is settled
}
# Ensemble: a function (backtesting/ensemble.py) turning the signal columns into one `side`, 1 / -1 while it holds (one
# signal alone: list just that one in SIGNALS). Longs and shorts run as separate strategies ({name}_long / _short).
# ensemble_function: in search mode lenient (long while any long signal is 1), else strict (while every one is 1);
# each trade's `signals` label says which fired
ENSEMBLE = ensemble_function
# Signals: False (normal) = where `side` turns 1 / -1 from 0 in its session, then no more until it goes back to 0; a
# trade exits by TP / SL / FLAT_AT or the opposite signal. True (search) = each new pattern of signals (the `signals`
# label) is a signal; a trade also exits at the next bar's open when its pattern changes, and after a TP / SL exit the
# next one waits for a new pattern
SEARCH_MODE = False
# No entry on a bar where any of these training / label columns is 1 (the entry bar, the one after the signal): news
# windows (FOMC / NFP / CPI / PPI / GDP). An open trade is kept; [] = no rule
NO_ENTRY = [f'news_{FREQ}']
# Flat when a bar with any of these columns 1 starts: an open trade exits at its first tick (result 0, like FLAT_AT),
# and no entry on it. [f'news_{FREQ}']: out of the market for every news window; [] = no rule (trades ride the news)
FLAT_ON = []
# No entry when the signal bar's VOL_COLUMN is below / above these quantiles of the previous VOL_SESSIONS sessions' bars
# (past only, no look-ahead; the first VOL_SESSIONS sessions do not trade): neither too calm nor too wild. None = no rule
VOL_BAND = (0.2, 0.8)
VOL_SESSIONS = 60

# Stops: a function (backtesting/stops.py) giving each signal its tp / sl (units); grid = every pair of units,
# vol_grid = every pair of multiples of the signal bar's VOL_COLUMN (wider when volatile, tighter when quiet)
STOP = grid(tp=[ 16, 24, 32,40,60,100], sl=[ 16, 24,32, 40,60,100])
# STOP = vol_grid(tp=[1, 2, 3, 4, 6], sl=[1, 2, 3, 4, 6], column=VOL_COLUMN)

# Sessions: 'YYYY-MM-DD', inclusive; None = from the first (after the burn-in) / to the last
START = None
END = None

# Costs and the session cut-off
POINT_VALUE = 12.5  # $ per price unit (ES)
COMMISSION = 1.9    # $ per side (entry and exit), so 3.8 a trade
SLIPPAGE = 1        # units per market fill, market runs only: the entry, a stop-loss, a signal or FLAT_AT exit (not a TP)
FLAT_AT = '22:58'   # shifted clock: every position is closed when this bar starts, before the next date
# Entry: 'market' at the entry bar's first tick (slippage), or 'limit' at its open with the order latency
# (stop_search.params.ENTRY_LATENCY_MS; backtesting/entry.py): filled at the open or better; skipped when it never
# fills or a TP / SL level is reached before the fill (needs the 1-min bar file's entry columns). A limit run uses
# limit orders throughout (the SL a stop-limit at its level, signal and FLAT_AT exits at the next bar's open): no slippage
ENTRY = 'market'

# Label analysis (python -m backtesting.analysis): trades before SPLIT pick a slice, trades from it test it
SPLIT = '2026-01-01'
# Training-data columns kept on each trade from its signal bar (known when the order goes in); the analysis labels
# them in n quantile buckets over all bars
# The FREQ bars' volatility (params.VOL_METHODS over params.WINDOW_SIZE bars, e.g. 20_std_1): 1 = calmest 20% .. 5
LABEL_FEATURES = {VOL_COLUMN: 5}
# The final run, reported with its statistics and charts in results/final/: (strategy name, stops); None = best net
FINAL = None  # e.g. ('ensemble_function', '40/24')

# ML (backtesting/ml.py, ml_ensemble.py, ml_stops.py): python -m backtesting.ml builds the features and the outcome
# table at FREQ (every bar long and short with every stop pair, exits by TP / SL / FLAT_AT only); then
# python -m backtesting.ml_ensemble and python -m backtesting.ml_stops each train walk-forward and write predictions
ML_STOP_MULTIPLES = [1, 2, 3, 4, 6]  # the outcome table's TP / SL pairs: multiples of VOL_COLUMN at the signal bar
ML_REFERENCE_STOPS = (2, 2)          # the ensemble's target: the net of this pair (independent of the stop model)
ML_THRESHOLD = 0.0                   # the ensemble takes a side when its predicted net per trade ($) is above this
ML_FIRST_TEST_YEAR = 2022            # walk-forward: each year from here is predicted by a model trained on the years before
ML_MAX_TRAIN = 2_000_000             # training rows sampled at most per fit (the stop model's table is large)
ML_PARAMS = {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 200, 'subsample': 0.8,
             'subsample_freq': 1, 'colsample_bytree': 0.8, 'verbose': -1}  # LightGBM
