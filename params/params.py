import datetime



NORM_FACTOR = 320 #{'1':50,'15':80,'30':100,'60':120,'720':300,'W':1500}

WINDOW_SIZE = 20

COMMISSION = 0

VOL_METHODS = 'std' # 'atr','aevdev

FEATURES = ['sma','sma_rsi','bbands','UD']#,'price_session','fvg']

#UNIT_TAS = ['std','atr','UD','sma_rsi','bbands']

TICK_DATA_PATH = '~/Desktop/projects/backtesting/data/processed/tick.dat'

DATA_PATH = '~/Desktop/projects/backtesting/data/processed'

FREQS = ['1','5','15','60','day']

BEGIN_DATE = datetime.datetime(2020, 1, 1, 0, 0, 0)

UD_PIVOTS = 5  # last U/D pivots kept per freq (data_pipeline/feature_engineering.py)





#1.25 or 12.5







