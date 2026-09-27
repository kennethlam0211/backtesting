"""
Backtest on the training data (data_pipeline/feature_engineering.py's output, one row per config.FREQ bar),
with stop_search deciding which of take-profit / stop-loss a trade hits first. Built step by step:

    1. read_training_data   the training parquet, burn-in dropped, optionally one date range
    2. find_exits           every signal row -> one trade: entry, exit, result and gross PnL
    3. one_at_a_time        signals while a trade is open are skipped
    4. add_costs            gross and net PnL (slippage and commission), in units and $
    5. run_grid             steps 2-4 for every stops of config.STOP (e.g. every TP x SL pair) on the same signals
    6. run_backtest         a strategy (backtesting/strategy.py) over the stops

    python -m backtesting.backtest          # config.SIGNALS through config.ENSEMBLE -> one table, results/trades.parquet
    python -m backtesting.analysis          # then: leaderboard, label analysis, the final run's statistics and charts

Prices and PnL are in units of price x4 (like the bar files and tick.dat): 1 unit = 0.25 index point.
The run (signals, ensemble, stops, sessions, costs) is set in backtesting/config.py; paths come from params/params.py.
"""
import datetime
import os
import time

import numpy as np
import polars as pl

from backtesting.config import (
    COMMISSION,
    END,
    ENSEMBLE,
    FLAT_AT,
    FREQ,
    LABEL_FEATURES,
    POINT_VALUE,
    SIGNALS,
    SLIPPAGE,
    START,
    STOP,
)
from backtesting.strategy import ensemble, expand
from params import (
    FREQS,
    TICK_DATA_PATH,
    TRADES_PATH,
    TRAINING_DATA_PATH,
    UD_PIVOTS,
    WINDOW_SIZE,
)
from stop_search import StopSearch

# Labels of the entry bar (when the trade is in the market), kept on each trade (when read) for analysis: session block
# (1 Asia, 2 Europe, 3 US), regular hours, hour of the shifted clock, news (1 inside an FOMC / NFP / CPI / PPI / GDP window)
LABEL_COLS = ['session', 'rth', 'hour', f'news_{FREQ}']


def _as_date(d):
    return d if d is None or isinstance(d, datetime.date) else datetime.date.fromisoformat(d)


def _seconds(freq: str) -> int:
    """A bar size in seconds: 'day', '<n>s' (seconds) or '<n>' (minutes)."""
    return 86400 if freq == 'day' else int(freq[:-1]) if freq.endswith('s') else int(freq) * 60


def _base_freq() -> str:
    """The smallest of params.FREQS: the training data has one row per such bar."""
    return min(FREQS, key=_seconds)


def _as_time(t):
    return t if isinstance(t, datetime.time) else datetime.time.fromisoformat(t)


def read_training_data(path=TRAINING_DATA_PATH, start=None, end=None, columns=None, freq=FREQ) -> pl.DataFrame:
    """
    Step 1: the training data as `freq` bars (default config.FREQ), one row per bar in time order, plus `date`, the
    session (the date of the shifted-clock ts). The training data has one row per base bar (the smallest of
    params.FREQS); for a larger `freq` the rows are its bars (_freq_bars: each bar's first appearance, the smaller
    timeframes' columns dropped, its start and labels from its bar file {freq}_ohlcv.parquet next to the training
    data, joined on start_ind_{freq}): the training data is the playground, any freq can be traded without rerunning
    feature engineering. `start_ind` is the traded bar's first tick in tick.dat.

    The burn-in is dropped as feature_engineering drops it: whole sessions, until one starts with every freq's
    UD_PIVOTS U/D pivots (so a file written with --keep-warmup reads the same as one without; the pivots never
    go back to empty once filled, so this only cuts sessions at the start).

    Args:
        path: the training parquet (default: feature_engineering's default output).
        start, end: first / last session date, inclusive (datetime.date or 'YYYY-MM-DD'); default: all.
        columns: columns to keep besides start_ind, ts and open_{freq}; default: every column.
        freq: the bar size traded, one of params.FREQS.

    Raises:
        ValueError: no row is left, or the bars are not in tick.dat order.
    """
    base = _base_freq()
    # Always read: the walk in find_exits needs the tick.dat row of every bar, the session and the entry price
    keys = [f'start_ind_{base}', 'ts', f'open_{base}', 'date']
    if freq != base:  # _freq_bars keeps each freq bar's first appearance and its open
        keys += [f'start_ind_{freq}', f'open_{freq}']
    start, end = _as_date(start), _as_date(end)
    pivots = [f'{WINDOW_SIZE}_UD_last{UD_PIVOTS}_{f}' for f in FREQS]
    # Pivots are NaN (Float64 files) or null (Int32 files) until filled; cast so both read the same
    ready = pl.all_horizontal([pl.col(c).cast(pl.Float64).fill_nan(None).is_not_null() for c in pivots])

    lf = pl.scan_parquet(path).with_columns(pl.col('ts').dt.date().alias('date'))
    lf = lf.filter(ready.first().over('date'))
    if start is not None:
        lf = lf.filter(pl.col('date') >= start)
    if end is not None:
        lf = lf.filter(pl.col('date') <= end)
    if columns is not None:
        own = _bar_cols(freq) if freq != base else []  # the freq's bar file gives these
        lf = lf.select(keys + [c for c in columns if c not in keys + own])
    df = lf.collect().rename({f'start_ind_{base}': 'start_ind'})
    if freq != base and not df.is_empty():
        df = _freq_bars(df, freq, os.path.join(os.path.dirname(path), f'{freq}_ohlcv.parquet'))

    if df.is_empty():
        raise ValueError(f"{path}: no rows after the burn-in between {start or 'the start'} and {end or 'the end'}")
    # find_exits walks bars by row: row order must be tick order
    if (df['start_ind'].diff().drop_nulls() <= 0).any():
        raise ValueError(f"{path}: start_ind is not strictly increasing")
    return df


def _bar_cols(freq: str) -> list[str]:
    """The columns _freq_bars takes from a `freq` bar file (read_training_data does not read them from the training data)."""
    return ['ts', 'session', 'rth', 'hour', f'news_{freq}', 'date']


def _freq_bars(rows: pl.DataFrame, freq: str, bar_path) -> pl.DataFrame:
    """
    One row per `freq` bar from the base rows:
    - each bar's first appearance (keep first on start_ind_{freq}): its last base bar, or the first later one when that
      one had no trades. The columns of every timeframe smaller than `freq` are dropped, so a row holds nothing after the
      bar's close (no bar of `freq` or larger ends inside a late appearance's extra minutes). `start_ind` becomes the
      bar's first tick, start_ind_{freq}.
    - ts (the bar's start), session, rth, hour and news_{freq} (any news flag in the bar) from its bar file
      (`bar_path`, step 2's {freq}_ohlcv.parquet), joined on the bar's first tick: exact for every bar, however late
      it appears.
    A bar of an earlier session (seen at the start of the rows) is left out.
    """
    key = f'start_ind_{freq}'
    smaller = tuple(f'_{f}' for f in FREQS if _seconds(f) < _seconds(freq))
    base_only = ['start_ind', 'ts', 'date', 'session', 'rth', 'hour']
    first = (rows.filter(pl.col(key).is_not_null())
             .unique(key, keep='first', maintain_order=True)
             .select([c for c in rows.columns if c not in base_only and not c.endswith(smaller)]))
    bars = (pl.read_parquet(bar_path)
            .select(pl.col('start_ind').alias(key), pl.col('ts').cast(pl.Datetime('ms')), 'session', 'rth', 'hour',
                    pl.any_horizontal(pl.col('^news_.*$') != 0).cast(pl.Int8).alias(f'news_{freq}'))
            .with_columns(date=pl.col('ts').dt.date()))
    out = first.join(bars, on=key, how='inner').filter(pl.col('date').is_in(rows['date'].unique().implode()))
    lead = ['ts', f'open_{freq}', 'session', 'rth', 'hour', f'news_{freq}', 'date']
    return out.sort(key).select(pl.col(key).alias('start_ind'), *lead, pl.exclude(key, *lead))


def find_exits(stops, df: pl.DataFrame, flat_at=FLAT_AT, stop=None, freq=FREQ, keep=()) -> pl.DataFrame:
    """
    Step 2: one trade per row of df whose `side` is 1 (long) or -1 (short); 0 or null = no trade.
    `tp` and `sl` (units, > 0) are the take-profit and stop-loss distances from the entry.

    The signal is known at its bar's close, so the trade enters at the first tick of the next FREQ bar of
    the same session. The bars are then checked one at a time with stops.first_hit_many, for every open trade
    at once, until a level is hit; a hit fills at the level itself (costs come in add_costs).

    A trade also exits on the opposite signal: on the first bar after its signal whose `side` is the other side,
    known at that bar's close, the trade exits at the next bar's first tick (a market fill).

    Always flat before the next date: a trade still open when the flat bar starts (the session's first bar that
    ends after `flat_at`, i.e. the one holding it; for 1-min bars the bar starting at `flat_at`) exits at that bar's
    first tick, and a signal whose entry would be at or after it is skipped. A session with no such bar (an early
    close) exits at its last tick.

    Args:
        stops: a StopSearch on the tick.dat the training data was built from.
        df: read_training_data's output plus the `side` column, and `tp` / `sl` when there is no `stop`.
        flat_at: time of the shifted clock ('HH:MM' or datetime.time), default config.FLAT_AT.
        stop: a stop function (backtesting/stops.py) giving the signal rows their `tp`, `sl` and `stops`, e.g.
            every (tp, sl) pair of a grid, all in one walk; None: df's own `tp` / `sl`.
        freq: the bar size of df's rows (read_training_data's `freq`), default config.FREQ.
        keep: more df columns each trade keeps from its signal bar.

    Returns:
        One row per trade, in signal order: `row` (the signal's row in df), `signal_ts` (the signal bar), `date`,
        `side`, `tp`, `sl`, `stops` (with a `stop`), the signal bar's config.LABEL_FEATURES that df has and `keep`
        (known when the order goes in), `entry_ts` (the entry bar: the trade's time), its `year` and `month`, the
        entry bar's LABEL_COLS that df has, `entry_ind` (tick.dat row), `entry_px`, `exit_row` (df row of the bar
        the trade exits in; for an opposite signal that signal's bar), `exit_ts` (when it exits), `exit_px`,
        `result` (1 take-profit, -1 stop-loss, 2 opposite signal, 0 flat at `flat_at` or the session end) and `pnl`
        (gross, units).

    Raises:
        ValueError: a signal has a missing or non-positive tp / sl, or tick.dat's price at an entry is not
            the training data's open_{FREQ} there (the two files come from different builds).
    """
    # Per session: the flat bar (the first at or after flat_at; null after an early close) and the stop row, the
    # first row a trade may not be in (the flat bar, else the row after the session's last bar)
    t = _as_time(flat_at)
    bar_end_minute = pl.col('ts').dt.hour().cast(pl.Int64) * 60 + pl.col('ts').dt.minute() + _seconds(freq) / 60
    flat_bar = pl.when(bar_end_minute > t.hour * 60 + t.minute).then(pl.col('row')).min().over('date')
    signal = pl.col('side').fill_null(0)
    next_row = lambda s: pl.when(signal == s).then(pl.col('row')).shift(-1).backward_fill()  # the next row with side s
    df = df.with_row_index('row').with_columns(_flat=flat_bar).with_columns(
        _stop=pl.col('_flat').fill_null(pl.col('row').max().over('date') + 1),
        # The opposite signal after each row: where a trade from it exits (none: past the last row)
        _change=pl.when(signal == 1).then(next_row(-1)).otherwise(next_row(1)).fill_null(pl.len()),
    )
    sig = df.filter((signal != 0) & (pl.col('row') + 1 < pl.col('_stop')))
    if stop is not None:
        sig = stop(sig)
        if 'stops' not in sig.columns:
            sig = sig.with_columns(stops=pl.lit(getattr(stop, '__name__', 'stop')))
    if sig.select((pl.col('tp').is_null() | pl.col('sl').is_null() | (pl.col('tp') <= 0) | (pl.col('sl') <= 0)).any()).item():
        raise ValueError("every signal needs tp and sl > 0 (units)")

    start = df['start_ind'].to_numpy()
    side = sig['side'].to_numpy().astype(np.int64)
    tp = sig['tp'].to_numpy()
    sl = sig['sl'].to_numpy()
    entry_row = sig['row'].to_numpy().astype(np.int64) + 1
    entry_px = stops.price(start[entry_row])
    open_px = df[f'open_{freq}'].to_numpy()[entry_row]
    if (entry_px != open_px).any():
        i = np.flatnonzero(entry_px != open_px)[0]
        raise ValueError(f"tick.dat price {entry_px[i]} at row {start[entry_row[i]]} is not open_{freq} {open_px[i]}: "
                         "training data and tick.dat come from different builds")

    upper = np.where(side == 1, entry_px + tp, entry_px + sl)
    lower = np.where(side == 1, entry_px - sl, entry_px - tp)

    # Walk: every open trade checks its current bar; a hit, or the last bar before its end row, closes it. The end
    # row: the session's stop row, or the bar after the opposite signal, whichever comes first
    session_stop = sig['_stop'].to_numpy().astype(np.int64)
    change = sig['_change'].to_numpy().astype(np.int64)
    until = np.minimum(session_stop, change + 1)
    bar = entry_row.copy()
    hit = np.zeros(len(sig), dtype=np.int8)
    todo = np.arange(len(sig))
    while todo.size:
        h = stops.first_hit_many(freq, start[bar[todo]], upper[todo], lower[todo])
        closed = (h != 0) | (bar[todo] + 1 == until[todo])
        hit[todo[closed]] = h[closed]
        todo = todo[~closed]
        bar[todo] += 1

    # A hit fills at its level (first_hit rounds a float upper up and a float lower down). No hit: an opposite signal
    # before the flat bar -> the first tick of the bar after it (bar + 1); else the flat bar's first tick, or
    # after an early close the session's last tick (the one before the row where its last bar ends)
    changed = (hit == 0) & (until < session_stop)
    change_px = stops.price(start[np.where(changed, bar + 1, 0)])
    flat = sig['_flat'].fill_null(-1).to_numpy().astype(np.int64)
    flat_px = stops.price(start[np.maximum(flat, 0)])
    close_px = stops.price(stops.bar_end(freq, start[bar]) - 1)
    time_px = np.where(flat >= 0, flat_px, close_px)
    exit_px = np.where(hit == 1, np.ceil(upper), np.where(hit == -1, np.floor(lower),
                                                           np.where(changed, change_px, time_px))).astype(np.int64)
    bar = np.where((hit == 0) & ~changed & (flat >= 0), flat, bar)

    ts = df['ts']
    labels = [c for c in LABEL_COLS if c in df.columns]
    features = [c for c in LABEL_FEATURES if c in sig.columns]
    run = ['stops'] if 'stops' in sig.columns else []
    return sig.select(['row', pl.col('ts').alias('signal_ts'), 'date', 'side', 'tp', 'sl', *run, *features, *keep]).with_columns(
        entry_ts=ts.gather(entry_row),
        year=ts.gather(entry_row).dt.year(),  # labels of the trade's time (entry_ts) for the analysis
        month=ts.gather(entry_row).dt.month(),
        **{c: df[c].gather(entry_row) for c in labels},  # the entry bar's labels
        entry_ind=pl.Series(start[entry_row]),
        entry_px=pl.Series(entry_px, dtype=pl.Int64),
        exit_row=pl.Series(bar, dtype=pl.UInt32),
        exit_ts=ts.gather(np.where(changed, bar + 1, bar)),
        exit_px=pl.Series(exit_px),
        result=pl.Series(np.where(changed, 2, hit * side), dtype=pl.Int8),
        pnl=pl.Series(side * (exit_px - entry_px)),
    )


def one_at_a_time(trades: pl.DataFrame) -> pl.DataFrame:
    """
    Step 3: one position at a time. Going through find_exits' trades in signal order, a signal while the previous
    kept trade is still open is skipped. A signal on the bar that trade exits in is taken: it is known at that bar's
    close, after the exit, and enters on the next bar. After an opposite signal that bar is the signal's own: it
    enters at the next bar's first tick, where the old trade exits (a reversal).
    """
    rows = trades['row'].to_numpy().astype(np.int64)
    exits = trades['exit_row'].to_numpy()
    if (np.diff(rows) < 0).any():
        raise ValueError("trades must be in signal order (a stop function must keep the signal rows' order)")
    keep = np.zeros(len(trades), dtype=bool)
    free_from = -1  # first row whose signal can be taken
    for i in range(len(trades)):
        if rows[i] >= free_from:
            keep[i] = True
            free_from = exits[i]
    return trades.filter(pl.Series(keep))


def add_costs(trades: pl.DataFrame, point_value=POINT_VALUE, commission=COMMISSION, slippage=SLIPPAGE) -> pl.DataFrame:
    """
    Step 4: gross and net PnL. Adds `gross_usd` (the gross `pnl` in $), `net_units` and `net_usd`.

    Slippage is `slippage` units on each market fill: the entry, and a stop-loss, opposite-signal or time (flat_at) exit. A take-profit
    is a resting limit order, filled at its level. The TP / SL levels stay measured from the entry's market price.
    Commission is per side, so twice a trade.
    """
    fills = pl.when(pl.col('result') == 1).then(1).otherwise(2)
    return trades.with_columns(
        gross_usd=pl.col('pnl') * point_value,
        net_units=pl.col('pnl') - slippage * fills,
    ).with_columns(
        net_usd=pl.col('net_units') * point_value - 2 * commission,
    )


def run_grid(stops, df: pl.DataFrame, stop=STOP, keep=()) -> pl.DataFrame:
    """
    Step 5: steps 2-4 for every `stops` the stop function gives the same signals (df has `side`), e.g. every TP x SL
    pair of a grid. Returns every stops' trades stacked; each is its own backtest, told apart by `stops`. All share
    one walk (find_exits with `stop`): far fewer, larger stop_search calls than one walk each.
    """
    trades = find_exits(stops, df, stop=stop, keep=keep)
    if trades.is_empty():
        return add_costs(trades)
    return pl.concat([add_costs(one_at_a_time(t)) for _, t in trades.group_by('stops', maintain_order=True)])


def run_backtest(strategy, stops, start=START, end=END, stop=STOP) -> pl.DataFrame:
    """
    Step 6: one strategy over every stops of `stop` (sessions and stops default to config.py): reads its columns,
    LABEL_COLS and config.LABEL_FEATURES, makes its signals and returns every stops' trades, with the strategy's
    name as `strategy` and its params as columns.

    Raises:
        ValueError: a column the strategy reads is not in the FREQ-bar frame (a freq below FREQ is dropped).
    """
    df = read_training_data(start=start, end=end, columns=[*strategy.columns, *LABEL_COLS, *LABEL_FEATURES])
    missing = [c for c in strategy.columns if c not in df.columns]
    if missing:
        raise ValueError(f"{strategy.name}: {missing} not in the {FREQ}-bar training data: a signal's freq must be "
                         f"config.FREQ ({FREQ}) or larger")
    df = strategy.signals(df)
    sessions = df['date'].unique().sort()
    print(f"{strategy.name}: {len(sessions):,} sessions ({sessions[0]} .. {sessions[-1]}), "
          f"{df.filter(pl.col('side').fill_null(0) != 0).height:,} signals")
    return run_grid(stops, df, stop, strategy.keep).select(
        pl.lit(strategy.name).alias('strategy'), *[pl.lit(v).alias(k) for k, v in strategy.params.items()], pl.all())


def main():
    """
    config.SIGNALS through config.ENSEMBLE as one strategy over config.STOP: all trades in one table,
    params.TRADES_PATH. One signal alone is an ensemble of one.
    """
    t0 = time.time()
    stops = StopSearch.load(TICK_DATA_PATH)
    trades = run_backtest(ensemble(expand(SIGNALS), ENSEMBLE), stops)
    os.makedirs(os.path.dirname(TRADES_PATH), exist_ok=True)
    trades.write_parquet(TRADES_PATH)
    print(f"{len(trades):,} trades of {trades.select('strategy', 'stops').n_unique()} runs in {TRADES_PATH} "
          f"({time.time() - t0:.0f}s); next: python -m backtesting.analysis")


if __name__ == "__main__":
    main()
