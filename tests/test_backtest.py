import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backtesting.backtest import find_exits, read_training_data
from data_pipeline.to_dat import offset_session, process_session
from params import FREQS, UD_PIVOTS, WINDOW_SIZE
from stop_search import DAT_COLS, StopSearch

PRICE_COL = DAT_COLS.index('price')
DATES = [datetime.date(2024, 3, 11), datetime.date(2024, 3, 12)]
PIVOTS = [f'{WINDOW_SIZE}_UD_last{UD_PIVOTS}_{f}' for f in FREQS]


def make_session(rng, date):
    """One session of step-1 ticks, as tests/test_stop_search.py: bursts and quiet gaps, price in ticks (x4)."""
    secs = np.cumsum(rng.exponential(rng.choice([0.1, 1.0, 20.0], size=40_000)))
    secs = secs[secs < 23 * 3600].astype(np.int64)
    n = len(secs)
    df = pd.DataFrame({
        'ts': pd.Timestamp(date) + pd.to_timedelta(secs, unit='s'),
        'price': 20_000 + np.cumsum(rng.integers(-1, 2, n)),
        'volume': rng.integers(1, 5, n),
    })
    df['ts'] = df['ts'].astype('datetime64[s]')
    for c in ['rth', 'session', 'hour', 'news_fomc', 'news_nfp', 'news_cpi', 'news_ppi', 'news_gdp']:
        df[c] = np.int8(0)
    return df


@pytest.fixture(scope='module')
def market():
    """tick.dat rows and the matching 1-min training rows (start_ind, ts, open_1, pivots) for two sessions."""
    rng = np.random.default_rng(0)
    blocks, bars, offset = [], [], 0
    for d in DATES:
        tick_res, res, n = process_session(make_session(rng, d), d)
        blocks.append(offset_session(tick_res, res, offset))
        bars.append(res['1'])
        offset += n
    b = np.concatenate(bars)
    df = pl.DataFrame({
        'start_ind': b['start_ind'],
        'ts': b['ts'].astype('datetime64[ms]'),  # parquet has no seconds unit: training data reads back in ms
        'open_1': b['open'],
        **{c: np.ones(len(b)) for c in PIVOTS},
    })
    return np.concatenate(blocks), df


@pytest.fixture(scope='module')
def stops(market):
    return StopSearch(market[0])


def training(market, tmp_path, warmup=0):
    """The training frame as step 1 reads it; the first `warmup` rows have no pivots yet (NaN)."""
    df = market[1].with_columns([pl.when(pl.int_range(pl.len()) < warmup).then(np.nan).otherwise(pl.col(c)).alias(c)
                                 for c in PIVOTS])
    path = tmp_path / 'training_data.parquet'
    df.write_parquet(path)
    return path


def tick_scan(data, df, row, side, tp, sl, flat_at):
    """
    Expected trade by walking every tick from the entry until the session's first bar at or after flat_at (or the
    session end when it has none): (result, exit_row, exit_px).
    """
    start = df['start_ind'].to_numpy()
    day = df.with_row_index('r').filter(pl.col('date') == df['date'][row])
    flat = day.filter(pl.col('ts').dt.time() >= flat_at)['r'].to_list()
    last = day['r'].max()
    end = start[flat[0]] if flat else data[start[last], DAT_COLS.index('next_ind_1')]
    e = start[row + 1]
    px = data[e, PRICE_COL]
    up, lo = (px + tp, px - sl) if side == 1 else (px + sl, px - tp)
    p = data[e:end, PRICE_COL]
    k = np.flatnonzero((p >= up) | (p <= lo))
    if len(k) == 0:
        # Flat at the flat bar's first tick, or the session's last tick after an early close
        return (0, flat[0], data[end, PRICE_COL]) if flat else (0, last, p[-1])
    first = k[0]
    exit_row = np.searchsorted(start, e + first, 'right') - 1
    return (1 if p[first] >= up else -1) * side, exit_row, up if p[first] >= up else lo


def test_read_drops_the_burn_in_by_whole_sessions(market, tmp_path):
    bars = market[1]
    day2 = bars['ts'].dt.date().to_list().index(DATES[1])  # first row of the second session
    # Pivots complete inside session 1: the whole session goes, as feature_engineering drops it
    df = read_training_data(training(market, tmp_path, warmup=100))
    assert df['start_ind'][0] == bars['start_ind'][day2]
    assert df['date'].unique().to_list() == [DATES[1]]
    # Complete exactly on session 2's first row: nothing more is cut
    df = read_training_data(training(market, tmp_path, warmup=day2))
    assert len(df) == len(bars) - day2


def test_read_date_range_and_columns(market, tmp_path):
    df = read_training_data(training(market, tmp_path), start='2024-03-12', columns=[])
    assert df.columns == ['start_ind', 'ts', 'open_1', 'date']
    assert df['date'].unique().to_list() == [DATES[1]]


def test_read_raises_when_nothing_is_left(market, tmp_path):
    with pytest.raises(ValueError, match='no rows after the burn-in'):
        read_training_data(training(market, tmp_path, warmup=len(market[1])))


# 22:59: the default; 12:00: many time exits; 23:30: no bar that late, as an early close
@pytest.mark.parametrize('flat_at', [datetime.time(22, 59), datetime.time(12, 0), datetime.time(23, 30)])
def test_exits_match_a_tick_scan(market, stops, tmp_path, flat_at):
    df = read_training_data(training(market, tmp_path))
    rng = np.random.default_rng(1)
    side = np.zeros(len(df), np.int8)
    pick = rng.choice(len(df), 400, replace=False)
    side[pick] = rng.choice([1, -1], len(pick))
    tp = rng.integers(1, 60, len(df))
    sl = rng.integers(1, 60, len(df))
    df = df.with_columns(side=pl.Series(side), tp=pl.Series(tp), sl=pl.Series(sl))

    trades = find_exits(stops, df, flat_at=flat_at)
    assert trades['result'].n_unique() == 3  # take-profits, stop-losses and time exits all covered
    assert (trades['entry_ts'].dt.time() < flat_at).all()  # no entry at or after flat_at
    for t in trades.iter_rows(named=True):
        result, exit_row, exit_px = tick_scan(market[0], df, t['row'], t['side'], t['tp'], t['sl'], flat_at)
        assert (t['result'], t['exit_row'], t['exit_px']) == (result, exit_row, exit_px), t
        assert t['pnl'] == t['side'] * (exit_px - t['entry_px'])


def test_no_entry_at_or_after_22_59(market, stops, tmp_path):
    df = read_training_data(training(market, tmp_path)).with_row_index('r')
    # Signals on the 22:58 bar (entry at 22:59) and on each session's last bar are skipped; row 0 is taken
    late = df.filter(pl.col('ts').dt.time() >= datetime.time(22, 58))['r'].to_list()
    assert late
    side = np.zeros(len(df), np.int8)
    side[late] = 1
    side[0] = 1
    trades = find_exits(stops, df.drop('r').with_columns(side=pl.Series(side), tp=pl.lit(40), sl=pl.lit(20)))
    assert trades['row'].to_list() == [0]


def test_bad_levels_and_a_different_build_raise(market, stops, tmp_path):
    df = read_training_data(training(market, tmp_path)).with_columns(side=pl.lit(1, pl.Int8), tp=pl.lit(40), sl=pl.lit(20))
    with pytest.raises(ValueError, match='tp and sl > 0'):
        find_exits(stops, df.with_columns(sl=pl.lit(0)))
    with pytest.raises(ValueError, match='different builds'):
        find_exits(stops, df.with_columns(pl.col('open_1') + 1))


def test_rsi_signals_fire_on_the_first_bar_into_a_zone():
    from backtesting.strategy import rsi_signals
    rsi = [50, 9, 5, 11, 8, 95, 91, 89, 92, 50]
    want = [0, 1, 0, 0, 1, -1, 0, 0, -1, 0]  # in, stay, out, in again; the same above 90
    df = pl.DataFrame({'20_sma_rsi_1': [1 - r / 50 for r in rsi]})  # the stored scale: -(RSI / 50 - 1)
    out = rsi_signals(df, low=10, high=90)
    assert out['side'].to_list() == want
    assert out['rsi_1'].to_list() == pytest.approx(rsi)


def test_rsi_strategy_and_the_configured_run():
    from backtesting import config
    from backtesting.strategy import Strategy, rsi
    s = rsi(10, 90)
    assert s.name == 'rsi_10_90' and s.columns == ('20_sma_rsi_1',)
    assert config.STRATEGIES and all(isinstance(s, Strategy) for s in config.STRATEGIES)


def test_pairs_in_one_walk_equal_one_walk_per_pair(market, stops, tmp_path):
    df = read_training_data(training(market, tmp_path))
    side = np.zeros(len(df), np.int8)
    side[np.random.default_rng(2).choice(len(df), 300, replace=False)] = 1
    df = df.with_columns(side=pl.Series(side) * np.where(np.arange(len(df)) % 2, 1, -1).astype(np.int8))
    pairs = [(4, 8), (16, 4), (40, 40)]
    one = find_exits(stops, df, pairs=pairs).sort('tp', 'sl', 'row')
    each = pl.concat([find_exits(stops, df.with_columns(tp=pl.lit(tp, pl.Int64), sl=pl.lit(sl, pl.Int64)))
                      for tp, sl in pairs]).sort('tp', 'sl', 'row')
    assert len(one) > 0 and one.equals(each)
