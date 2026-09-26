import datetime



#NORM_FACTOR = 320 #{'1':50,'15':80,'30':100,'60':120,'720':300,'W':1500}

WINDOW_SIZE = 20


VOL_METHODS = 'std' # 'atr','aevdev

FEATURES = ['sma','sma_rsi','bbands','UD']#,'price_session','fvg']

#UNIT_TAS = ['std','atr','UD','sma_rsi','bbands']

# Paths, relative to the repo root: every script runs from there (python -m ...). Keep them relative: tests
DATA_PATH = 'data/processed'                               # step 2 output (to_dat): tick.dat + {freq}_ohlcv.parquet
TICK_DATA_PATH = f'{DATA_PATH}/tick.dat'                   # StopSearch.load() default
TRAINING_DATA_PATH = f'{DATA_PATH}/training_data.parquet'  # feature_engineering output = backtest inpu
RESULTS_DIR = 'results'                                    # backtest results
TRADES_PATH = f'{RESULTS_DIR}/trades.parquet'              # every trade of the last backtest run: the analysis input

FREQS = ['1','5','15','60','day']

BEGIN_DATE = datetime.datetime(2020, 1, 1, 0, 0, 0)

UD_PIVOTS = 5  # last U/D pivots kept per freq (data_pipeline/feature_engineering.py)



# Backtest settings (strategy, grid, costs): backtesting/config.py







