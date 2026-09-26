"""
Backtest on the training data (data_pipeline/feature_engineering.py's output, one row per 1-min bar),
with stop_search deciding which of take-profit / stop-loss a trade hits first. Built step by step:

    1. read_training_data   the training parquet, burn-in dropped, optionally one date range
    2. find_exits           every signal row -> one trade: entry, exit, result and gross PnL
    3. one_at_a_time        signals while a trade is open are skipped
    4. add_costs            gross and net PnL (slippage and commission), in units and $
    5. run_grid             steps 2-4 for every TP x SL pair on the same signals
    6. run_backtest         a strategy (backtesting/strategy.py) over the grid

    python -m backtesting.backtest          # every strategy in backtesting/config.py -> one table, results/trades.parquet
    python -m backtesting.analysis          # then: leaderboard, label analysis, the final run's statistics and charts

Prices and PnL are in units of price x4 (like the bar files and tick.dat): 1 unit = 0.25 index point.
The run (strategy, sessions, grid, costs) is set in backtesting/config.py; paths come from params/params.py.
"""
import datetime
import os
import time

import numpy as np
import polars as pl

from backtesting.config import (
    COMMISSION,
    END,
    FLAT_AT,
    POINT_VALUE,
    SL_GRID,
    SLIPPAGE,
    START,
    STRATEGIES,
    TP_GRID,
)
from params import (
    FREQS,
    TICK_DATA_PATH,
    TRADES_PATH,
    TRAINING_DATA_PATH,
    UD_PIVOTS,
    WINDOW_SIZE,
)
from stop_search import StopSearch

# Always read: the walk in find_exits needs the tick.dat row of every bar, the session and the entry price
BASE_COLS = ['start_ind', 'ts', 'open_1']
# Labels of the entry bar (when the trade is in the market), kept on each trade (when read) for analysis: session block
# (1 Asia, 2 Europe, 3 US), regular hours, hour of the shifted clock, news (1 inside an FOMC / NFP / CPI / PPI / GDP window)
LABEL_COLS = ['session', 'rth', 'hour', 'news_1']


def _as_date(d):
    return d if d is None or isinstance(d, datetime.date) else datetime.date.fromisoformat(d)


def _as_time(t):
    return t if isinstance(t, datetime.time) else datetime.time.fromisoformat(t)


def read_training_data(path=TRAINING_DATA_PATH, start=None, end=None, columns=None) -> pl.DataFrame:
    """
    Step 1: the training data, one row per 1-min bar in time order, plus `date`, the session (the date of the
    shifted-clock ts).

    The burn-in is dropped as feature_engineering drops it: whole sessions, until one starts with every freq's
    UD_PIVOTS U/D pivots (so a file written with --keep-warmup reads the same as one without; the pivots never
    go back to empty once filled, so this only cuts sessions at the start).

    Args:
        path: the training parquet (default: feature_engineering's default output).
        start, end: first / last session date, inclusive (datetime.date or 'YYYY-MM-DD'); default: all.
        columns: feature columns to keep besides start_ind, ts and open_1; default: every column.

    Raises:
        ValueError: no row is left, or the bars are not in tick.dat order.
    """
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
        lf = lf.select(BASE_COLS + ['date'] + [c for c in columns if c not in BASE_COLS + ['date']])
    df = lf.collect()

    if df.is_empty():
        raise ValueError(f"{path}: no rows after the burn-in between {start or 'the start'} and {end or 'the end'}")
    # find_exits walks bars by row: row order must be tick order
    if (df['start_ind'].diff().drop_nulls() <= 0).any():
        raise ValueError(f"{path}: start_ind is not strictly increasing")
    return df


def find_exits(stops, df: pl.DataFrame, flat_at=FLAT_AT, pairs=None) -> pl.DataFrame:
    """
    Step 2: one trade per row of df whose `side` is 1 (long) or -1 (short); 0 or null = no trade.
    `tp` and `sl` (units, > 0) are the take-profit and stop-loss distances from the entry.

    The signal is known at its bar's close, so the trade enters at the first tick of the next 1-min bar of
    the same session. The bars are then checked one at a time with stops.first_hit_many, for every open trade
    at once, until a level is hit; a hit fills at the level itself (costs come in add_costs).

    Always flat before the next date: a trade still open when the session's first bar at or after `flat_at`
    starts exits at that bar's first tick, and a signal whose entry would be at or after it is skipped. A
    session with no bar from `flat_at` on (an early close) exits at its last tick.

    Args:
        stops: a StopSearch on the tick.dat the training data was built from.
        df: read_training_data's output plus the `side`, `tp` and `sl` columns.
        flat_at: time of the shifted clock ('HH:MM' or datetime.time), default config.FLAT_AT.
        pairs: (tp, sl) pairs to run every signal with, all in one walk, instead of df's `tp` / `sl` columns;
            the trades then come signal by signal, each signal's pairs in this order.

    Returns:
        One row per trade, in signal order: `row` (the signal's row in df), `signal_ts` (the signal bar), `date`,
        `side`, `tp`, `sl`, `entry_ts` (the entry bar: the trade's time), the entry bar's LABEL_COLS that df has,
        `entry_ind` (tick.dat row), `entry_px`, `exit_row` (df row of the bar the trade exits in), `exit_ts`,
        `exit_px`, `result` (1 take-profit, -1 stop-loss, 0 flat at `flat_at` or the session end) and `pnl`
        (gross, units).

    Raises:
        ValueError: a signal has a missing or non-positive tp / sl, or tick.dat's price at an entry is not
            the training data's open_1 there (the two files come from different builds).
    """
    # Per session: the flat bar (the first at or after flat_at; null after an early close) and the stop row, the
    # first row a trade may not be in (the flat bar, else the row after the session's last bar)
    flat_bar = pl.when(pl.col('ts').dt.time() >= _as_time(flat_at)).then(pl.col('row')).min().over('date')
    df = df.with_row_index('row').with_columns(_flat=flat_bar).with_columns(
        _stop=pl.col('_flat').fill_null(pl.col('row').max().over('date') + 1))
    sig = df.filter((pl.col('side').fill_null(0) != 0) & (pl.col('row') + 1 < pl.col('_stop')))
    if pairs is not None:
        levels = pl.DataFrame(list(pairs), schema={'tp': pl.Int64, 'sl': pl.Int64}, orient='row')
        sig = sig.drop(['tp', 'sl'], strict=False).join(levels, how='cross')
    if sig.select((pl.col('tp').is_null() | pl.col('sl').is_null() | (pl.col('tp') <= 0) | (pl.col('sl') <= 0)).any()).item():
        raise ValueError("every signal needs tp and sl > 0 (units)")

    start = df['start_ind'].to_numpy()
    side = sig['side'].to_numpy().astype(np.int64)
    tp = sig['tp'].to_numpy()
    sl = sig['sl'].to_numpy()
    entry_row = sig['row'].to_numpy().astype(np.int64) + 1
    entry_px = stops.price(start[entry_row])
    open_1 = df['open_1'].to_numpy()[entry_row]
    if (entry_px != open_1).any():
        i = np.flatnonzero(entry_px != open_1)[0]
        raise ValueError(f"tick.dat price {entry_px[i]} at row {start[entry_row[i]]} is not open_1 {open_1[i]}: "
                         "training data and tick.dat come from different builds")

    upper = np.where(side == 1, entry_px + tp, entry_px + sl)
    lower = np.where(side == 1, entry_px - sl, entry_px - tp)

    # Walk: every open trade checks its current bar; a hit, or the last bar before the stop row, closes it
    stop = df['_stop'].to_numpy().astype(np.int64)[entry_row]
    bar = entry_row.copy()
    hit = np.zeros(len(sig), dtype=np.int8)
    todo = np.arange(len(sig))
    while todo.size:
        h = stops.first_hit_many('1', start[bar[todo]], upper[todo], lower[todo])
        closed = (h != 0) | (bar[todo] + 1 == stop[todo])
        hit[todo[closed]] = h[closed]
        todo = todo[~closed]
        bar[todo] += 1

    # A hit fills at its level (first_hit rounds a float upper up and a float lower down). No hit: the flat bar's
    # first tick, or after an early close the session's last tick (the one before the row where its last bar ends)
    flat = df['_flat'].fill_null(-1).to_numpy().astype(np.int64)[entry_row]
    flat_px = stops.price(start[np.maximum(flat, 0)])
    close_px = stops.price(stops.bar_end('1', start[bar]) - 1)
    time_px = np.where(flat >= 0, flat_px, close_px)
    exit_px = np.where(hit == 1, np.ceil(upper), np.where(hit == -1, np.floor(lower), time_px)).astype(np.int64)
    bar = np.where((hit == 0) & (flat >= 0), flat, bar)

    ts = df['ts']
    labels = [c for c in LABEL_COLS if c in df.columns]
    return sig.select(['row', pl.col('ts').alias('signal_ts'), 'date', 'side', 'tp', 'sl']).with_columns(
        entry_ts=ts.gather(entry_row),
        **{c: df[c].gather(entry_row) for c in labels},  # the entry bar's labels
        entry_ind=pl.Series(start[entry_row]),
        entry_px=pl.Series(entry_px, dtype=pl.Int64),
        exit_row=pl.Series(bar, dtype=pl.UInt32),
        exit_ts=ts.gather(bar),
        exit_px=pl.Series(exit_px),
        result=pl.Series(hit * side, dtype=pl.Int8),
        pnl=pl.Series(side * (exit_px - entry_px)),
    )


def one_at_a_time(trades: pl.DataFrame) -> pl.DataFrame:
    """
    Step 3: one position at a time. Going through find_exits' trades in signal order, a signal while the previous
    kept trade is still open is skipped. A signal on the bar that trade exits in is taken: it is known at that bar's
    close, after the exit, and enters on the next bar.
    """
    rows = trades['row'].to_numpy()
    exits = trades['exit_row'].to_numpy()
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

    Slippage is `slippage` units on each market fill: the entry, and a stop-loss or time (flat_at) exit. A take-profit
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


def run_grid(stops, df: pl.DataFrame, tp_grid=TP_GRID, sl_grid=SL_GRID) -> pl.DataFrame:
    """
    Step 5: steps 2-4 for every (tp, sl) pair on the same signals (df has `side`). Returns every pair's trades stacked;
    each pair is its own backtest, told apart by the `tp` / `sl` columns. All pairs share one walk (find_exits
    with `pairs`): far fewer, larger stop_search calls than one walk per pair.
    """
    trades = find_exits(stops, df, pairs=[(tp, sl) for tp in tp_grid for sl in sl_grid])
    return pl.concat([add_costs(one_at_a_time(t)) for _, t in trades.group_by(['tp', 'sl'], maintain_order=True)])


def run_backtest(strategy, stops, start=START, end=END, tp_grid=TP_GRID, sl_grid=SL_GRID) -> pl.DataFrame:
    """
    Step 6: one strategy over every TP x SL pair (sessions, grid and costs default to config.py): reads its columns
    and LABEL_COLS, makes its signals and returns every pair's trades, with the strategy's name as `strategy`.
    """
    df = strategy.signals(read_training_data(start=start, end=end, columns=[*strategy.columns, *LABEL_COLS]))
    sessions = df['date'].unique().sort()
    print(f"{strategy.name}: {len(sessions):,} sessions ({sessions[0]} .. {sessions[-1]}), "
          f"{df.filter(pl.col('side') != 0).height:,} signals, {len(tp_grid) * len(sl_grid)} TP x SL pairs")
    return run_grid(stops, df, tp_grid, sl_grid).select(pl.lit(strategy.name).alias('strategy'), pl.all())


def main():
    """Every strategy in config.STRATEGIES over the TP x SL grid: all their trades in one table, params.TRADES_PATH."""
    t0 = time.time()
    stops = StopSearch.load(TICK_DATA_PATH)
    trades = pl.concat([run_backtest(s, stops) for s in STRATEGIES])
    os.makedirs(os.path.dirname(TRADES_PATH), exist_ok=True)
    trades.write_parquet(TRADES_PATH)
    print(f"{len(trades):,} trades of {trades.select('strategy', 'tp', 'sl').n_unique()} runs in {TRADES_PATH} "
          f"({time.time() - t0:.0f}s); next: python -m backtesting.analysis")


if __name__ == "__main__":
    main()
