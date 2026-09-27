import datetime

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from backtesting.backtest import find_exits, read_training_data
from data_pipeline.to_dat import bars_table, offset_session, process_session
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


def appears(b5):
    """When each 5-min bar shows up in the 1-min table, as feature_engineering joins it: on its last minute."""
    ts = b5['ts'].astype('datetime64[ms]')
    close = np.minimum(ts + np.timedelta64(5, 'm'), ts.astype('datetime64[D]') + np.timedelta64(23, 'h'))
    return close - np.timedelta64(1, 'm')


@pytest.fixture(scope='module')
def market():
    """
    tick.dat rows, the matching 1-min training rows (start_ind, ts, open_1, pivots, and the 5-min start_ind_5 / open_5
    joined as feature_engineering does) and the 5-min bars (per session, as step 2 writes them) for two sessions.
    """
    rng = np.random.default_rng(0)
    blocks, bars, bars5, offset = [], [], [], 0
    for d in DATES:
        tick_res, res, n = process_session(make_session(rng, d), d)
        blocks.append(offset_session(tick_res, res, offset))
        bars.append(res['1'])
        bars5.append(res['5'])
        offset += n
    b = np.concatenate(bars)
    df = pl.DataFrame({
        'start_ind': b['start_ind'],
        'ts': b['ts'].astype('datetime64[ms]'),  # parquet has no seconds unit: training data reads back in ms
        'open_1': b['open'],
        **{c: np.ones(len(b)) for c in PIVOTS},
    })
    b5 = np.concatenate(bars5)
    df = df.join_asof(pl.DataFrame({'_ts': appears(b5), 'start_ind_5': b5['start_ind'], 'open_5': b5['open']}),
                      left_on='ts', right_on='_ts', strategy='backward').drop('_ts')
    return np.concatenate(blocks), df, bars5


@pytest.fixture(scope='module')
def stops(market):
    return StopSearch(market[0])


def training(market, tmp_path, warmup=0):
    """The training frame as step 1 reads it; the first `warmup` rows have no pivots yet (NaN)."""
    df = market[1].with_columns([pl.when(pl.int_range(pl.len()) < warmup).then(np.nan).otherwise(pl.col(c)).alias(c)
                                 for c in PIVOTS])
    path = tmp_path / 'training_data.parquet'
    df.rename({'start_ind': 'start_ind_1'}).write_parquet(path)  # the file suffixes every freq's start_ind
    pq.write_table(bars_table(market[2]), tmp_path / '5_ohlcv.parquet')  # its bar file next to it, as step 2 writes it
    return path


def tick_scan(data, df, row, side, tp, sl, flat_at, freq='1'):
    """
    Expected trade by walking every tick from the entry until the session's first `freq` bar that ends after flat_at
    (or the session end when it has none), or until the bar after df's next opposite `side` when that comes first:
    (result, exit_row, exit_px).
    """
    start = df['start_ind'].to_numpy()
    day = df.with_row_index('r').filter(pl.col('date') == df['date'][row])
    minute = day['ts'].dt.hour().cast(pl.Int64) * 60 + day['ts'].dt.minute()
    flat = day.filter(minute + int(freq) > flat_at.hour * 60 + flat_at.minute)['r'].to_list()
    last = day['r'].max()
    end = start[flat[0]] if flat else data[start[last], DAT_COLS.index(f'next_ind_{freq}')]
    opposite = [r for r in np.flatnonzero(df['side'].to_numpy() == -side) if r > row]
    if opposite and opposite[0] + 1 < (flat[0] if flat else last + 1):
        k = opposite[0]
        end = start[k + 1]
    else:
        k = None
    e = start[row + 1]
    px = data[e, PRICE_COL]
    up, lo = (px + tp, px - sl) if side == 1 else (px + sl, px - tp)
    p = data[e:end, PRICE_COL]
    hits = np.flatnonzero((p >= up) | (p <= lo))
    if len(hits) == 0:
        # The next bar's first tick after an opposite signal; else flat at the flat bar's first tick, or the
        # session's last tick after an early close
        if k is not None:
            return 2, k, data[end, PRICE_COL]
        return (0, flat[0], data[end, PRICE_COL]) if flat else (0, last, p[-1])
    first = hits[0]
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
    assert trades['result'].n_unique() == 4  # take-profits, stop-losses, opposite signals and time exits all covered
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
    trades = find_exits(stops, df.drop('r').with_columns(side=pl.Series(side), tp=pl.lit(40), sl=pl.lit(20)),
                        flat_at=datetime.time(22, 59))
    assert trades['row'].to_list() == [0]


def test_bad_levels_and_a_different_build_raise(market, stops, tmp_path):
    df = read_training_data(training(market, tmp_path)).with_columns(side=pl.lit(1, pl.Int8), tp=pl.lit(40), sl=pl.lit(20))
    with pytest.raises(ValueError, match='tp and sl > 0'):
        find_exits(stops, df.with_columns(sl=pl.lit(0)))
    with pytest.raises(ValueError, match='different builds'):
        find_exits(stops, df.with_columns(pl.col('open_1') + 1))


def test_rsi_long_and_short_are_states():
    from backtesting.strategy import rsi_long, rsi_short
    rsi = [50, 9, 5, 11, 8, 95, 91, 89, 92, 50]
    df = pl.DataFrame({'20_sma_rsi_1': [1 - r / 50 for r in rsi]})  # the stored scale: -(RSI / 50 - 1)
    long, short = rsi_long(freq='1', level=10), rsi_short(freq='1', level=90)
    assert long.signals(df)['side'].to_list() == [0, 1, 1, 0, 1, 0, 0, 0, 0, 0]  # every bar below 10
    assert short.signals(df)['side'].to_list() == [0, 0, 0, 0, 0, -1, -1, 0, -1, 0]  # every bar above 90
    assert (long.name, short.name) == ('rsi_long_1_10', 'rsi_short_1_90')
    assert long.columns == ('20_sma_rsi_1',) and long.params == {'freq': '1', 'level': 10}


def test_expand_and_the_configured_run():
    from backtesting import config
    from backtesting.strategy import Strategy, expand, rsi_long
    got = expand({rsi_long: {'freq': ['1', '5'], 'params': {'level': [5, 10]}}})
    assert [s.name for s in got] == ['rsi_long_1_5', 'rsi_long_1_10', 'rsi_long_5_5', 'rsi_long_5_10']
    assert got[3].columns == ('20_sma_rsi_5',)
    configured = expand(config.SIGNALS)
    assert configured and all(isinstance(s, Strategy) for s in configured)
    assert len({s.name for s in configured}) == len(configured)
    assert callable(config.ENSEMBLE) and callable(config.STOP)


def test_ensemble_function_votes_among_the_active_signals():
    from backtesting.ensemble import ensemble_function
    rows = [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],        # 3 of 3 long: 1
            [1, 1, -1, -1, 1, 0, 0, 0, 0, 0],      # +1 of 5 active (< 30%): 0
            [-1, -1, -1, 1, 0, 0, 0, 0, 0, 0],     # -2 of 4: -1
            [0] * 10,                              # none active: 0
            [1, -1, 0, 0, 0, 0, 0, 0, 0, 0],       # 0 of 2: 0
            [1, 1, 1, -1, -1, -1, -1, -1, -1, -1],  # -4 of 10: -1
            [1, 1, 1, 1, -1, -1, -1, -1, -1, -1]]   # -2 of 10 (> -30%): 0
    names = [f's{i}' for i in range(10)]
    df = pl.DataFrame(rows, schema={n: pl.Int8 for n in names}, orient='row')
    assert df.select(side=ensemble_function(df, names))['side'].to_list() == [1, 0, -1, 0, 0, -1, 0]


def test_ensemble_side_is_the_trigger_bar_only():
    from backtesting.ensemble import ensemble_function
    from backtesting.strategy import ensemble, rsi_long, rsi_short
    rsi = [50, 9, 5, 8, 50, 9, 95, 91, 50]
    df = pl.DataFrame({'20_sma_rsi_1': [1 - r / 50 for r in rsi]})
    e = ensemble([rsi_long(freq='1', level=10), rsi_short(freq='1', level=90)], ensemble_function)
    out = e.signals(df)
    assert out['rsi_long_1_10'].to_list() == [0, 1, 1, 1, 0, 1, 0, 0, 0]  # the signals stay states
    assert out['rsi_short_1_90'].to_list() == [0, 0, 0, 0, 0, 0, -1, -1, 0]
    assert out['side'].to_list() == [0, 1, 0, 0, 0, 1, -1, 0, 0]  # the vote only where it turns 1 / -1
    assert (e.name, e.columns, e.keep) == ('ensemble_function', ('20_sma_rsi_1',), ('rsi_long_1_10', 'rsi_short_1_90'))


def test_opposite_signal_exits_at_the_next_open_and_reverses(market, stops, tmp_path):
    from backtesting.backtest import add_costs, one_at_a_time
    df = read_training_data(training(market, tmp_path))
    side = np.zeros(len(df), np.int8)
    side[[100, 103]] = 1   # the second long is skipped: the first is still open
    side[105] = -1         # the opposite signal: the long exits at row 106's first tick, the short enters there
    df = df.with_columns(side=pl.Series(side), tp=pl.lit(10_000), sl=pl.lit(10_000))  # levels never hit
    trades = add_costs(one_at_a_time(find_exits(stops, df)))
    start = df['start_ind'].to_numpy()
    long, short = trades.row(0, named=True), trades.row(1, named=True)
    assert trades['row'].to_list()[:2] == [100, 105]
    assert (long['result'], long['exit_row'], long['exit_ts']) == (2, 105, df['ts'][106])
    assert long['exit_px'] == stops.price(start[106]) == short['entry_px']
    assert long['net_units'] == long['pnl'] - 2  # a market exit: slippage on both fills


def test_grid_in_one_walk_equals_one_walk_per_pair(market, stops, tmp_path):
    from backtesting.stops import grid
    df = read_training_data(training(market, tmp_path))
    side = np.zeros(len(df), np.int8)
    side[np.random.default_rng(2).choice(len(df), 300, replace=False)] = 1
    df = df.with_columns(side=pl.Series(side) * np.where(np.arange(len(df)) % 2, 1, -1).astype(np.int8))
    one = find_exits(stops, df, stop=grid(tp=[4, 16], sl=[8, 40])).sort('tp', 'sl', 'row')
    each = pl.concat([find_exits(stops, df.with_columns(tp=pl.lit(tp, pl.Int64), sl=pl.lit(sl, pl.Int64)))
                      .with_columns(stops=pl.lit(f'{tp}/{sl}')) for tp in [4, 16] for sl in [8, 40]])
    assert len(one) > 0 and one.equals(each.select(one.columns).sort('tp', 'sl', 'row'))


def test_a_stop_function_is_its_own_run(market, stops, tmp_path):
    from backtesting.stops import grid
    df = read_training_data(training(market, tmp_path))
    side = np.zeros(len(df), np.int8)
    side[np.random.default_rng(4).choice(len(df), 200, replace=False)] = 1
    df = df.with_columns(side=pl.Series(side))

    def fixed(signals):
        return signals.with_columns(tp=pl.lit(8, pl.Int64), sl=pl.lit(4, pl.Int64))

    mine = find_exits(stops, df, stop=fixed)
    assert mine['stops'].unique().to_list() == ['fixed']
    assert mine.drop('stops').equals(find_exits(stops, df, stop=grid(tp=[8], sl=[4])).drop('stops'))


def test_freq_5_rows_are_each_bars_first_appearance(market, tmp_path):
    path = training(market, tmp_path)
    pl.read_parquet(path).with_columns(x=pl.int_range(pl.len())).write_parquet(path)  # numbers the 1-min rows
    df = read_training_data(path, freq='5')
    b5 = np.concatenate(market[2])
    # A bar's row is the first 1-min row showing it: its last minute, or later when that minute had no trades
    first = np.searchsorted(market[1]['ts'].to_numpy(), appears(b5), 'left')
    seen = first < len(market[1])
    assert (~seen).sum() <= 1  # only the data's very last bar can go unseen (no minute after its last one)
    assert df['start_ind'].to_list() == b5['start_ind'][seen].tolist()
    assert df['open_5'].to_list() == b5['open'][seen].tolist()
    assert df['ts'].to_list() == pl.Series(b5['ts'][seen].astype('datetime64[ms]')).to_list()  # the bar's own start
    assert df['x'].to_list() == first[seen].tolist()
    # The late case is covered: some bars' last minute had no trades, so they first show a minute or more later
    assert (market[1]['ts'].to_numpy()[first[seen]] > appears(b5)[seen]).sum() > 10
    assert df.columns[:8] == ['start_ind', 'ts', 'open_5', 'session', 'rth', 'hour', 'news_5', 'date']
    assert not [c for c in df.columns if c.endswith('_1')]  # the 1-min columns are dropped


# 22:59: the flat bar is the 22:55 bar, the one holding 22:59; 12:02: the 12:00 bar
@pytest.mark.parametrize('flat_at', [datetime.time(22, 59), datetime.time(12, 2)])
def test_freq_5_exits_match_a_tick_scan(market, stops, tmp_path, flat_at):
    df = read_training_data(training(market, tmp_path), freq='5')
    rng = np.random.default_rng(3)
    side = np.zeros(len(df), np.int8)
    pick = rng.choice(len(df), 150, replace=False)
    side[pick] = rng.choice([1, -1], len(pick))
    df = df.with_columns(side=pl.Series(side), tp=pl.Series(rng.integers(1, 60, len(df))),
                         sl=pl.Series(rng.integers(1, 60, len(df))))
    trades = find_exits(stops, df, flat_at=flat_at, freq='5')
    assert len(trades) > 0
    for t in trades.iter_rows(named=True):
        result, exit_row, exit_px = tick_scan(market[0], df, t['row'], t['side'], t['tp'], t['sl'], flat_at, freq='5')
        assert (t['result'], t['exit_row'], t['exit_px']) == (result, exit_row, exit_px), t
