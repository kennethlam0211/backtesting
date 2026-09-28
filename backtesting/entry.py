"""
Limit entries with order latency, from the 1-min bar file's entry columns (stop_search.params.ENTRY_COLS, written by
data_pipeline/to_dat.py), no tick data needed: a limit order at a minute's open reaches the market ENTRY_LATENCY_MS
after the minute starts. It fills at once when price is then at the open or better (at that price), else when price
comes back to the open within the minute (at the open). The trade is let go when it never fills, or when price reached
the take-profit or the stop-loss level (measured from the open) during the latency or while the order waited.
"""
import os

import numpy as np
import polars as pl

from params import DATA_PATH
from stop_search.params import ENTRY_COLS

ENTRIES_PATH = os.path.join(DATA_PATH, '1_ohlcv.parquet')  # the 1-min bar file next to tick.dat
_SIDE_COLS = {side: [f'{c}_{side}_1' for c in ('fill_px', 'after_high', 'after_low')] for side in ('long', 'short')}


class Entries:
    """The 1-min bars' open and entry columns, looked up by each minute's first tick.dat row (start_ind)."""

    def __init__(self, bars: pl.DataFrame):
        missing = [c for c in ('start_ind', 'open', *ENTRY_COLS) if c not in bars.columns]
        if missing:
            raise ValueError(f"the 1-min bars have no entry columns {missing}: rebuild them with data_pipeline/to_dat.py")
        bars = bars.sort('start_ind')
        self.start = bars['start_ind'].to_numpy()
        # Per minute: open, pre_high, pre_low, then the long's fill / after_high / after_low and the short's
        self.values = bars.select('open', 'pre_high_1', 'pre_low_1', *_SIDE_COLS['long'], *_SIDE_COLS['short']).to_numpy()

    @classmethod
    def load(cls, path=ENTRIES_PATH):
        return cls(pl.read_parquet(path, columns=['start_ind', 'open', *ENTRY_COLS]))


_default = {}


def default_entries() -> Entries:
    """The 1-min bar file next to tick.dat (params.DATA_PATH), loaded once."""
    if 'entries' not in _default:
        _default['entries'] = Entries.load()
    return _default['entries']


def get_entry_many(entries: Entries, starts, side, tp, sl) -> np.ndarray:
    """
    Limit entries at the open of the minutes starting at tick.dat rows `starts` (e.g. an entry bar's start_ind: its
    first minute), all at once. `side` 1 (buy) / -1 (sell); `tp` / `sl` the take-profit / stop-loss distances (units)
    from the open, single values or one per entry.

    Returns the entry price per entry: a better price than the open when price was there as the order arrived, else the
    open when price came back to it. 0 when the entry fails and the trade is skipped: price reached the take-profit or
    the stop-loss level during the latency (pre_high / pre_low) or while the order waited (after_high / after_low), or
    never came back to the open within the minute. A level counts as reached as in stops.first_hit: price >= the upper
    level or <= the lower one. Before the fill no level was reached, so the exits can be walked from the minute's start.

    Raises:
        ValueError: a row is not the first tick of a 1-min bar, or a side is not 1 / -1.
    """
    starts = np.asarray(starts, dtype=np.int64)
    side = np.asarray(side)
    if starts.ndim != 1 or np.broadcast_shapes(side.shape, starts.shape) != starts.shape:
        raise ValueError("starts must be 1-D and side a single value or one per start")
    if not np.isin(side, (1, -1)).all():
        raise ValueError("side must be 1 (buy) or -1 (sell)")
    at = np.minimum(np.searchsorted(entries.start, starts), len(entries.start) - 1)
    not_start = entries.start[at] != starts
    if not_start.any():
        raise ValueError(f"{not_start.sum()} rows are not the first tick of a 1-min bar, e.g. row {starts[not_start][0]}")
    rows = entries.values[at]
    long = np.broadcast_to(side == 1, starts.shape)
    open_px, pre_high, pre_low = rows[:, 0], rows[:, 1], rows[:, 2]
    fill, after_high, after_low = (np.where(long, rows[:, 3 + k], rows[:, 6 + k]) for k in range(3))
    tp, sl = np.asarray(tp), np.asarray(sl)
    upper = np.where(long, open_px + tp, open_px + sl)
    lower = np.where(long, open_px - sl, open_px - tp)
    ok = (fill > 0) & (pre_high < upper) & (pre_low > lower) & (after_high < upper) & (after_low > lower)
    return np.where(ok, fill, 0).astype(np.int64)
