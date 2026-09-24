import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd
from search import search, col_names, col_to_ind_dict

def build_test_array(prices, ts_start="2023-01-01 09:30:00", tick_ms=1000):
    """
    Given a list of prices, build a synthetic numpy array following the layout.
    """
    n = len(prices)
    data = np.zeros((n, len(col_names)), dtype=np.int64)
    ts = pd.date_range(ts_start, periods=n, freq=f"{tick_ms}ms")

    data[:, col_to_ind_dict['price']] = prices
    data[:, col_to_ind_dict['ts']] = ts.astype('int64')

    # helper for frequencies
    df = pd.DataFrame({'price': prices}, index=ts)

    freq_map = {
        '1': '1min', '2': '2min', '3': '3min',
        '4': '4min', '5': '5min', '10': '10min', '15': '15min',
        '20': '20min', '30': '30min', '60': '60min', 'day': 'D', '1s': '1s', '15s': '15s'
    }
    offset_map = {}

    # Calculate for each freq
    for f_name, f_str in freq_map.items():
        offset = offset_map.get(f_name, pd.Timedelta(0))
        # Groupby
        grouper = pd.Grouper(freq=f_str, offset=offset)
        grouped = df.groupby(grouper)

        for name, group in grouped:
            if len(group) == 0:
                continue
            start_idx = df.index.get_loc(group.index[0])
            end_idx = df.index.get_loc(group.index[-1])

            high = group['price'].max()
            low = group['price'].min()
            next_ind = end_idx + 1

            data[start_idx, col_to_ind_dict[f'high_{f_name}']] = high
            data[start_idx, col_to_ind_dict[f'low_{f_name}']] = low
            data[start_idx, col_to_ind_dict[f'next_ind_{f_name}']] = next_ind

            if f_name == '1':
                data[start_idx, col_to_ind_dict['start_ind']] = start_idx

    return data

def test_search_basic():
    prices = [100, 101, 102, 105, 103, 98, 95]
    # ticks are 1s apart, so all in one 1-min bar
    data = build_test_array(prices)

    # target_high hit first
    idx, side = search(data, 0, 104, 96)
    assert side == 1
    assert idx == 3
    assert data[idx, col_to_ind_dict['price']] == 105

    # target_low hit first
    idx, side = search(data, 0, 106, 99)
    assert side == -1
    assert idx == 5
    assert data[idx, col_to_ind_dict['price']] == 98

    # entry mid-bar (already crossed a level before entry!)
    # prices = [100, 101, 102, 105, 103, 98, 95]
    # at start_idx=4, price is 103. target_high=104 was hit at idx=3, but we start at 4!
    idx, side = search(data, 4, 104, 99)
    # The lowest after idx=4 is 95 at idx=6. So it should hit 99 at idx=5.
    assert side == -1
    assert idx == 5

def test_search_exact_hit():
    prices = [100, 101, 102, 103, 104]
    data = build_test_array(prices)

    idx, side = search(data, 0, 102, 90)
    assert side == 1
    assert idx == 2

def test_search_never_hit():
    prices = [100, 101, 102, 101, 100]
    data = build_test_array(prices)

    idx, side = search(data, 0, 105, 95)
    assert idx is None
    assert side is None

def test_search_last_tick():
    prices = [100, 101, 102, 101, 105]
    data = build_test_array(prices)

    idx, side = search(data, 0, 104, 90)
    assert side == 1
    assert idx == 4

def test_search_bar_boundary():
    # Construct a dataset that spans multiple 1-min bars
    # 0s, 30s, 60s, 90s, 120s
    prices = [100, 101, 102, 103, 104, 105]
    data = build_test_array(prices, tick_ms=30000) # 30 sec per tick
    # indices:
    # 0: 0s (1min bar 1 start)
    # 1: 30s
    # 2: 60s (1min bar 2 start)
    # 3: 90s
    # 4: 120s (1min bar 3 start)
    # 5: 150s

    idx, side = search(data, 0, 103, 90)
    assert side == 1
    assert idx == 3

def brute_force_search(data, start_idx, target_high, target_low):
    price_col = col_to_ind_dict['price']
    for i in range(start_idx, len(data)):
        if data[i, price_col] >= target_high:
            return i, 1
        if data[i, price_col] <= target_low:
            return i, -1
    return None, None

def test_search_random_property():
    np.random.seed(42)
    # Random walk
    for _ in range(50):
        n = np.random.randint(100, 3000)
        steps = np.random.choice([-1, 0, 1], size=n)
        prices = 1000 + np.cumsum(steps)
        # Random tick intervals (1s to 500s) to create gaps and multiple sessions
        gaps = np.random.exponential(np.random.choice([0.5, 5, 60]), size=n)  # dense and sparse ticks
        # Add big gaps to simulate day changes
        for idx in np.random.choice(range(10, n), size=3):
            gaps[idx] += 3600 * 16 # 16 hours gap

        ts_start = pd.Timestamp("2023-01-01 09:30:00")
        ts_vals = ts_start + pd.to_timedelta(np.cumsum(gaps), unit='s')

        data = np.zeros((n, len(col_names)), dtype=np.int64)
        data[:, col_to_ind_dict['price']] = prices
        data[:, col_to_ind_dict['ts']] = ts_vals.astype('int64')

        df = pd.DataFrame({'price': prices}, index=ts_vals)

        freq_map = {
            '1': '1min', '2': '2min', '3': '3min',
            '4': '4min', '5': '5min', '10': '10min', '15': '15min',
            '20': '20min', '30': '30min', '60': '60min', 'day': 'D', '1s': '1s', '15s': '15s'
        }
        offset_map = {}

        for f_name, f_str in freq_map.items():
            offset = offset_map.get(f_name, pd.Timedelta(0))
            grouper = pd.Grouper(freq=f_str, offset=offset)
            grouped = df.groupby(grouper)
            for name, group in grouped:
                if len(group) == 0: continue
                start_idx = df.index.get_loc(group.index[0])
                end_idx = df.index.get_loc(group.index[-1])
                data[start_idx, col_to_ind_dict[f'high_{f_name}']] = group['price'].max()
                data[start_idx, col_to_ind_dict[f'low_{f_name}']] = group['price'].min()
                data[start_idx, col_to_ind_dict[f'next_ind_{f_name}']] = end_idx + 1

        # Now test random queries
        for _ in range(10):
            start_idx = np.random.randint(0, n)
            start_price = prices[start_idx]
            th = start_price + np.random.randint(1, 20)
            tl = start_price - np.random.randint(1, 20)

            bf_idx, bf_side = brute_force_search(data, start_idx, th, tl)
            fast_idx, fast_side = search(data, start_idx, th, tl)

            assert fast_idx == bf_idx
            assert fast_side == bf_side

if __name__ == "__main__":
    pytest.main(["-v", "test_search.py"])
