import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import pyarrow as pa
import numpy as np
import datetime
from raw_data_preprocessing import to_ticks, SHIFT

def create_mock_table(ts_events, prices, sizes):
    """Creates a raw parquet-like pyarrow table."""
    return pa.Table.from_arrays(
        [
            pa.array(pd.Series(ts_events).dt.tz_localize("UTC")),
            pa.array(prices, type=pa.float64()),
            pa.array(sizes, type=pa.int64()),
        ],
        names=['ts_event', 'price', 'size']
    )

def test_news_flags_exact_bounds():
    # FOMC is at 14:00 ET = 20:00:00 New York wall clock + 6h
    # NFP is at 08:30 ET = 14:30:00 New York wall clock + 6h

    # We mock a session on 2024-09-18 (FOMC day)
    # The actual window for 14:00 is [13:55, 14:05) ET
    # Mapped to our ts (+6h shift): [19:55:00, 20:05:00)

    # Let's create some raw events
    # To get `ts = X` we need `ts_event = X - 6h` in NY time, then to UTC.
    # Actually, we can just create the UTC times directly.
    # ET is America/New_York. In Sept, EDT is UTC-4.
    # So 14:00 ET is 18:00 UTC.
    # Window in UTC is [17:55, 18:05) UTC.

    times_utc = [
        "2024-09-18 17:54:59.999", # Before window (ts: 19:54:59)
        "2024-09-18 17:55:00.000", # Exactly at start (ts: 19:55:00) -> FLAG
        "2024-09-18 18:00:00.000", # Middle (ts: 20:00:00) -> FLAG
        "2024-09-18 18:04:59.999", # Just before end (ts: 20:04:59) -> FLAG
        "2024-09-18 18:05:00.000", # Exactly at end (ts: 20:05:00) -> NO FLAG
    ]

    ts_events = pd.to_datetime(times_utc)
    prices = [1.0] * len(ts_events)
    sizes = [1] * len(ts_events)

    table = create_mock_table(ts_events, prices, sizes)

    session_date = pd.Timestamp("2024-09-18")

    out_table = to_ticks(table, session_date, "ESZ4")
    df = out_table.to_pandas()

    assert len(df) == 5

    fomc_flags = df['news_fomc'].tolist()
    assert fomc_flags == [0, 1, 1, 1, 0]

    # Others should be 0
    assert (df['news_nfp'] == 0).all()
    assert (df['news_cpi'] == 0).all()

def test_news_flags_two_events_same_time():
    # CPI and PPI can be at 08:30 on the same day? The yaml has them separate, but we test the logic
    # In the mock, we can just inject a fake event or use a known one. Let's monkeypatch NEWS_EVENTS.
    from raw_data_preprocessing import NEWS_EVENTS

    # Fake session: 2024-01-01
    # We will fake a CPI and PPI event on this day at 08:30 ET
    # 08:30 ET is 13:30 UTC in Jan (EST is UTC-5).
    # Window: 08:25 to 08:35 ET
    # Our ts will be: 14:25 to 14:35

    fake_start = np.array([pd.Timestamp("2024-01-01 14:25:00").to_datetime64()])
    fake_end = np.array([pd.Timestamp("2024-01-01 14:35:00").to_datetime64()])

    NEWS_EVENTS['cpi'] = (fake_start, fake_end)
    NEWS_EVENTS['ppi'] = (fake_start, fake_end)

    times_utc = [
        "2024-01-01 13:24:59.000", # Before
        "2024-01-01 13:25:00.000", # In
        "2024-01-01 13:35:00.000", # Out
    ]

    ts_events = pd.to_datetime(times_utc)
    table = create_mock_table(ts_events, [1,1,1], [1,1,1])

    session_date = pd.Timestamp("2024-01-01")

    out_table = to_ticks(table, session_date, "ESH4")
    df = out_table.to_pandas()

    assert df['news_cpi'].tolist() == [0, 1, 0]
    assert df['news_ppi'].tolist() == [0, 1, 0]
    assert df['news_fomc'].tolist() == [0, 0, 0]

def test_news_flags_cross_midnight():
    # In our ts, if an event was at 17:58 ET (which doesn't happen for these, but testing logic)
    # The window would cross midnight since 18:00 ET is our 00:00 (next day in our session logic).
    # Wait, our ts is just wall + 6h.
    # 17:58 ET + 6h = 23:58
    # Window + 5m = 00:03 the next day.
    # But our session is exactly [00:00, 23:00). So it wouldn't cross midnight in our session (it would cross the gap).
    # Still, let's test a window that overlaps the session bounds.
    from raw_data_preprocessing import NEWS_EVENTS

    # Fake event at 22:58 ts (which is 16:58 ET). Window is [22:53, 23:03)
    fake_start = np.array([pd.Timestamp("2024-01-02 22:53:00").to_datetime64()])
    fake_end = np.array([pd.Timestamp("2024-01-02 23:03:00").to_datetime64()])
    NEWS_EVENTS['gdp'] = (fake_start, fake_end)

    # Events at end of session
    times_utc = [
        "2024-01-02 21:50:00.000", # 22:50 ts -> 0
        "2024-01-02 21:55:00.000", # 22:55 ts -> 1
        "2024-01-02 21:59:59.000", # 22:59:59 ts -> 1 (last tick of session)
    ]
    ts_events = pd.to_datetime(times_utc)
    table = create_mock_table(ts_events, [1,1,1], [1,1,1])
    session_date = pd.Timestamp("2024-01-02")

    out_table = to_ticks(table, session_date, "ESH4")
    df = out_table.to_pandas()

    assert df['news_gdp'].tolist() == [0, 1, 1]

def test_no_events():
    # Session with no events
    times_utc = [
        "2024-01-03 12:00:00.000",
    ]
    ts_events = pd.to_datetime(times_utc)
    table = create_mock_table(ts_events, [1], [1])
    session_date = pd.Timestamp("2024-01-03")

    out_table = to_ticks(table, session_date, "ESH4")
    df = out_table.to_pandas()

    assert df['news_fomc'].tolist() == [0]
    assert df['news_nfp'].tolist() == [0]
