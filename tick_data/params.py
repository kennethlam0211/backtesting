# Bar sizes built by to_dat.py. Second bars are appended last so every other column keeps its
# position from the HSI layout.
FREQS = ["1", "5", "10", "15", "30", "60", "day", "1s", "15s"]
# Bar sizes also written as OHLCV parquet
PARQUET_FREQS = ["1", "5", "10", "15", "30", "60", "day"]

# tick.dat column order (int64, row-major); readers must use this list, the file carries no header
DAT_COLS = ['start_ind', 'ts', 'price'] + [c for f in FREQS for c in (f'high_{f}', f'low_{f}', f'next_ind_{f}')]

# Each bar size -> the next smaller size that divides it evenly. Bars are aligned to the session open,
# so a bar's first row is also the first row of its first child, and its children tile it exactly.
# None: a 1s bar holding both levels is walked tick by tick.
CHILD = {'day': '60', '60': '30', '30': '10', '10': '5', '15': '5',
         '5': '1', '1': '15s', '15s': '1s', '1s': None}

# Where to_dat.py writes tick.dat by default
DAT_PATH = 'data/zarr/tick.dat'
