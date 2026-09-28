"""The entry columns data_pipeline/to_dat.py writes per 1-min bar: a limit order at the open, 300 ms latency."""
import datetime

import numpy as np
import pandas as pd

from data_pipeline.to_dat import offset_session, process_session
from stop_search.params import ENTRY_COLS, ENTRY_LATENCY_MS

DAY = datetime.date(2024, 3, 11)
DAY0 = pd.Timestamp(DAY).value


def session(seconds, prices):
    """A step-1 session from tick times (seconds after the session start, sub-second) and prices."""
    ts_ns = DAY0 + (np.asarray(seconds) * 1e9).round().astype(np.int64)
    n = len(prices)
    return pd.DataFrame({
        'ts': ts_ns.astype('datetime64[ns]').astype('datetime64[s]'), 'price': np.asarray(prices, np.int32),
        'volume': np.ones(n, np.int64), 'rth': np.zeros(n, np.int8), 'session': np.ones(n, np.int8),
        'hour': np.zeros(n, np.int8), **{c: np.zeros(n, np.int8) for c in ['news_fomc', 'news_nfp', 'news_cpi',
                                                                          'news_ppi', 'news_gdp']},
        'ts_ns': ts_ns,
    })


def entry_columns(df, offset=0):
    """The 1-min bars' entry columns as the bar file stores them, one dict per minute."""
    tick_res, res, _ = process_session(df, DAY)
    offset_session(tick_res, res, offset)
    return [dict(zip(ENTRY_COLS, (int(v) for v in r))) for r in res['1'][ENTRY_COLS]]


def reference(seconds, prices):
    """Tick by tick: per minute, the latency window, then the buy and the sell limit at the open."""
    seconds, prices = np.asarray(seconds), np.asarray(prices)
    out = []
    for m in np.unique((seconds // 60).astype(int)):
        p = prices[(seconds // 60).astype(int) == m]
        s = seconds[(seconds // 60).astype(int) == m]
        a = int(np.argmax(s >= m * 60 + ENTRY_LATENCY_MS / 1000)) if (s >= m * 60 + ENTRY_LATENCY_MS / 1000).any() else len(p)
        pre = p[:a] if a > 0 else p[:1]
        row = {'pre_high_1': int(pre.max()), 'pre_low_1': int(pre.min())}
        for side in ('long', 'short'):
            better = (lambda x, o=p[0]: x <= o) if side == 'long' else (lambda x, o=p[0]: x >= o)
            if a == len(p):
                fill, after = 0, p[:1]
            elif better(p[a]):
                fill, after = int(p[a]), p[a:a + 1]
            else:
                back = [t for t in range(a + 1, len(p)) if better(p[t])]
                fill, after = (int(p[0]), p[a:back[0]]) if back else (0, p[a:])
            row.update({f'fill_px_{side}_1': fill, f'after_high_{side}_1': int(after.max()),
                        f'after_low_{side}_1': int(after.min())})
        out.append(row)
    return out


def test_hand_worked_minutes():
    seconds = [0.100, 0.200, 0.350, 0.500, 0.600, 30.0,     # minute 0: open 100
               60.050, 60.400, 61.0, 70.0,                 # minute 1: open 200, price runs up and stays up
               120.500, 121.0]                             # minute 2: the first trade comes after the 300 ms
    prices = [100, 101, 102, 101, 100, 99, 200, 203, 205, 204, 300, 299]
    got = entry_columns(session(seconds, prices))
    # Minute 0: during the latency 100..101. At 0.35 s price is 102: worse for the buy, which waits (102, 101) and
    # fills at the open 100 at 0.6 s; better for the sell, which fills at once at 102
    assert got[0] == {'pre_high_1': 101, 'pre_low_1': 100,
                      'fill_px_long_1': 100, 'after_high_long_1': 102, 'after_low_long_1': 101,
                      'fill_px_short_1': 102, 'after_high_short_1': 102, 'after_low_short_1': 102}
    # Minute 1: the buy never gets back to 200 (203..205 while it waits); the sell fills at once at 203
    assert got[1] == {'pre_high_1': 200, 'pre_low_1': 200,
                      'fill_px_long_1': 0, 'after_high_long_1': 205, 'after_low_long_1': 203,
                      'fill_px_short_1': 203, 'after_high_short_1': 203, 'after_low_short_1': 203}
    # Minute 2: no trade during the latency; the open itself is the first trade after it: both fill at 300
    assert got[2] == {'pre_high_1': 300, 'pre_low_1': 300,
                      'fill_px_long_1': 300, 'after_high_long_1': 300, 'after_low_long_1': 300,
                      'fill_px_short_1': 300, 'after_high_short_1': 300, 'after_low_short_1': 300}
    assert got == reference(seconds, prices)


def test_random_sessions_match_a_tick_scan():
    rng = np.random.default_rng(7)
    for _ in range(5):
        seconds = np.sort(rng.uniform(0, 3 * 3600, 20_000))
        prices = 20_000 + np.cumsum(rng.integers(-1, 2, len(seconds)))
        got, want = entry_columns(session(seconds, prices)), reference(seconds, prices)
        assert got == want
        fills = np.array([[r['fill_px_long_1'], r['fill_px_short_1']] for r in got])
        assert (fills == 0).any() and (fills > 0).any()  # both cases covered


def test_whole_second_files_still_work():
    df = session([0.1, 0.5, 30.0], [100, 101, 100]).drop(columns='ts_ns')  # an older step-1 file: no ts_ns
    got = entry_columns(df)
    # Every trade in second 0 counts as during the latency (100..101); the first one the order meets is at 30 s: 100
    assert (got[0]['pre_high_1'], got[0]['pre_low_1'], got[0]['fill_px_long_1']) == (101, 100, 100)


def test_limit_entry_ignores_a_touch_before_the_order_arrives():
    import polars as pl

    from backtesting.backtest import find_exits
    from backtesting.entry import Entries
    from stop_search import StopSearch
    # Minute 0 signals a buy; minute 1 opens at 200 and spikes to 206 at 0.1 s, during the latency: a TP of 5 (205) is
    # hit before the order is in, so the limit entry lets the trade go
    seconds = [0.0, 30.0, 60.05, 60.10, 60.20, 60.40, 61.0, 90.0, 120.0, 150.0, 180.0]
    prices = [100, 101, 200, 206, 203, 200, 201, 202, 199, 198, 200]
    tick_res, res, _ = process_session(session(seconds, prices), DAY)
    stops = StopSearch(offset_session(tick_res, res, 0))
    b = res['1']
    df = pl.DataFrame({'start_ind': b['start_ind'], 'ts': b['ts'].astype('datetime64[ms]'), 'open_1': b['open'],
                       'side': [1, 0, 0, 0], 'tp': [5] * 4, 'sl': [5] * 4}).with_columns(date=pl.col('ts').dt.date())
    entries = Entries(pl.DataFrame({name: b[name] for name in ['start_ind', 'open', *ENTRY_COLS]}))
    limit = find_exits(stops, df, search_mode=False, entry='limit', freq='1', entries=entries)
    assert limit.is_empty()
    market = find_exits(stops, df, search_mode=False, entry='market', freq='1')
    assert market['result'].to_list() == [1]  # a market entry at the first tick is in for the spike
