import datetime



NORM_FACTOR = 320 #{'1':50,'15':80,'30':100,'60':120,'720':300,'W':1500}

WINDOW_SIZE = 20#40

COMMISSION = 0.6

WINDOWS = [20] # 1minute >>60 mins

FEATURES = ['fvg','std','atr','sma_rsi','sma','bbands','price_session']

UNIT_TAS = ['std','atr','UD','sma_rsi','bbands','bband_targets']

TICK_DATA_PATH = '~/Desktop/projects/backtesting/data/processed/tick.dat'

DATA_PATH = '~/Desktop/projects/backtesting/data/processed'

FREQS = ['1','15','60','day']

BEGIN_DATE = datetime.datetime(2023, 1, 1, 0, 0, 0)













