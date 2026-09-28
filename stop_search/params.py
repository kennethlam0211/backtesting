# tick.dat that StopSearch.load() opens when no path is given, set in params/params.py (the repo's settings).
# Only this name is imported: params.FREQS (the feature freqs) is not the FREQS below.
import numpy as np

from params import TICK_DATA_PATH

DAT_PATH = TICK_DATA_PATH

# Bar sizes built by data_pipeline/to_dat.py. Second bars are appended last so every other column keeps its
# position from the HSI layout.
FREQS = ["1", "5", "10", "15", "30", "60", "day", "1s", "15s", "5s"]
# Bar sizes also written as OHLCV parquet. 'session': one bar per session block of each date (step 1's `session`:
# Asia 00-08, Europe 08-16, US 16-23 on the shifted clock); a bar file only, not in tick.dat
PARQUET_FREQS = ["1", "5", "10", "15", "30", "60", "day", "session"]

# Limit entry at a 1-min bar's open: columns of the 1-min bar file (data_pipeline/to_dat.py). The order reaches the market
# ENTRY_LATENCY_MS after the minute starts. pre_high / pre_low: the highest / lowest trade during the latency (the open
# when there is none). Per side: fill_px, the first trade from then if it is at the open or better (at or below it for a
# buy, at or above for a sell); if it is worse, the open when price comes back to it within the minute; else 0.
# after_high / after_low: the highest / lowest trade from then until that return (the fill itself when it is at once;
# to the minute's end when price never comes back)
ENTRY_LATENCY_MS = 300
ENTRY_COLS = ['pre_high_1', 'pre_low_1'] + [f'{c}_{side}_1' for side in ('long', 'short')
                                            for c in ('fill_px', 'after_high', 'after_low')]

# tick.dat's columns (the file carries no header: readers must use DAT_DTYPE). Each row is ROW_BYTES of packed
# little-endian fields: first the uint32 columns (INDEX_COLS: row numbers, and ts in seconds, good until 2106), then
# the uint16 ones (PRICE_COLS: prices in ticks, x4, at most 65,535 = 16,383.75 points), then 2 bytes of padding so
# every row starts on a 4-byte boundary. Read as a table of each width, they are the two views StopSearch walks.
INDEX_COLS = ['start_ind', 'ts'] + [f'next_ind_{f}' for f in FREQS]
PRICE_COLS = ['price'] + [c for f in FREQS for c in (f'high_{f}', f'low_{f}')]
DAT_COLS = INDEX_COLS + PRICE_COLS
# The padding is a named field (_pad, always 0): unnamed padding is dropped by np.concatenate, which would make rows
# 2 bytes shorter
DAT_DTYPE = np.dtype([*((c, '<u4') for c in INDEX_COLS), *((c, '<u2') for c in PRICE_COLS), ('_pad', '<u2')])
ROW_BYTES = DAT_DTYPE.itemsize  # 12 x 4 + 21 x 2 + 2 = 92

# Each bar size -> the next smaller size that divides it evenly. Bars are aligned to the session open,
# so a bar's first row is also the first row of its first child, and its children tile it exactly.
# None: a 1s bar holding both levels is walked tick by tick.
CHILD = {'day': '60', '60': '30', '30': '10', '10': '5', '15': '5',
         '5': '1', '1': '15s', '15s': '5s', '5s': '1s', '1s': None}
