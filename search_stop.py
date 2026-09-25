import os

import numpy as np
from numba import njit, prange

from to_dat import DAT_COLS

# Each bar size -> the next smaller size that divides it evenly. Bars are aligned to the session open,
# so a bar's first row is also the first row of its first child, and its children tile it exactly.
# None: a 1s bar holding both levels is walked tick by tick.
CHILD = {'day': '60', '60': '30', '30': '10', '10': '5', '15': '5',
         '5': '1', '1': '15s', '15s': '1s', '1s': None}


def _seconds(freq):
    if freq == 'day':
        return 24 * 3600
    return int(freq[:-1]) if freq.endswith('s') else int(freq) * 60


for _parent, _child in CHILD.items():
    assert _child is None or _seconds(_parent) % _seconds(_child) == 0, f"{_child} bars do not tile {_parent} bars"

_IX = {c: k for k, c in enumerate(DAT_COLS)}
PRICE_COL = _IX['price']


def _chain(freq):
    """(next_ind, high, low) columns from `freq` down to 1s, following CHILD."""
    rows = []
    while freq is not None:
        rows.append((_IX[f'next_ind_{freq}'], _IX[f'high_{freq}'], _IX[f'low_{freq}']))
        freq = CHILD[freq]
    return np.array(rows, dtype=np.int64)


CHAINS = {f: _chain(f) for f in CHILD}

_INT = (int, np.integer)

# Kernel result when the bar summaries contradict the ticks. Returned rather than raised:
# an exception inside a prange loop is dropped silently.
BROKEN = 2


@njit(cache=True)
def _search_stop(data, ind, chain, price_col, target_high, target_low):
    depth = chain.shape[0]
    ends = np.empty(depth, dtype=np.int64)  # ends[k]: end of the parent bar whose children level k scans
    ends[0] = data[ind, chain[0, 0]]
    k = 0
    while True:
        if ind >= ends[k]:
            return BROKEN  # a bar held a level but none of its sub-bars did
        nxt = data[ind, chain[k, 0]]
        if nxt == 0:
            return BROKEN  # sub-bar missing
        hit_high = data[ind, chain[k, 1]] >= target_high
        hit_low = data[ind, chain[k, 2]] <= target_low
        if hit_high != hit_low:
            return 1 if hit_high else -1
        if not hit_high:
            if k == 0:
                return 0  # nothing inside the entry bar: skip
            ind = nxt  # next bar of the same size, still inside the parent
        elif k + 1 < depth:
            k += 1  # both inside: split into the child bars, which start on the same row
            ends[k] = nxt
        else:
            for t in range(ind, nxt):  # both inside a 1s bar: walk its ticks
                price = data[t, price_col]
                if price >= target_high:
                    return 1
                if price <= target_low:
                    return -1
            return BROKEN  # a 1s bar held a level but none of its ticks did


def search_stop(data, target_high, target_low, freq, start_idx):
    """
    Which level each entry's `freq` bar touches first, looking only inside that bar.

    Takes one entry or many: ints give an int back; 1-D arrays give an int8 array back, one value
    per entry in input order. Scalars broadcast against arrays (e.g. one level for every entry).
    Many entries run in one compiled call across all CPU cores (NUMBA_NUM_THREADS caps it).

    Same logic as search_org.search_stop: one level inside a bar decides it; both inside splits the bar
    into its CHILD bars, checked in time order; a 1s bar with both inside is walked tick by tick.

    Args:
        data: tick.dat as an int64 array (see load_dat).
        target_high: upper level(s), price x100 (4000.25 -> 400025); hit when price >= target_high.
        target_low: lower level(s), price x100; hit when price <= target_low.
        freq: bar size to look inside, shared by all entries: 'day', '60', '30', '15', '10', '5',
            '1', '15s' or '1s'.
        start_idx: entry row(s); each must be the first tick of a `freq` bar.

    Returns:
         1  target_high is hit first (a long's take-profit, a short's stop)
        -1  target_low is hit first (a long's stop, a short's take-profit)
         0  neither is hit inside the bar (skip)
        An int for one entry, an int8 array for many.

    Raises:
        ValueError: a start row is not the first tick of a `freq` bar, the inputs do not broadcast
            to one 1-D shape, or the bar summaries contradict the ticks.

    Example:
        import numpy as np
        from search_stop import search_stop, load_dat, PRICE_COL
        from to_dat import DAT_COLS

        data = load_dat('data/zarr/tick.dat')
        starts = np.flatnonzero(data[:, DAT_COLS.index('next_ind_1')])  # first tick of every 1-min bar
        entry = data[starts, PRICE_COL]

        # one entry: +10 pts / -5 pts inside the first minute
        side = search_stop(data, entry[0] + 1000, entry[0] - 500, '1', starts[0])
        # side == 1: +10 came first; -1: -5 came first; 0: neither within that minute

        # every minute at once, same offsets from each entry price
        sides = search_stop(data, entry + 1000, entry - 500, '1', starts)
        print((sides == 1).mean(), (sides == -1).mean(), (sides == 0).mean())  # share of each outcome
    """
    chain = CHAINS[freq]
    if isinstance(start_idx, _INT) and isinstance(target_high, _INT) and isinstance(target_low, _INT):
        start_idx = int(start_idx)
        if data[start_idx, chain[0, 0]] == 0:
            raise ValueError(f"row {start_idx} is not the first tick of a {freq} bar")
        side = int(_search_stop(data, start_idx, chain, PRICE_COL, int(target_high), int(target_low)))
        if side == BROKEN:
            raise ValueError(f"bar summaries of the {freq} bar at row {start_idx} contradict its ticks")
        return side

    try:
        starts, highs, lows = np.broadcast_arrays(*(np.asarray(a, dtype=np.int64) for a in (start_idx, target_high, target_low)))
    except ValueError:
        raise ValueError("start_idx, target_high and target_low must broadcast to one 1-D shape") from None
    if starts.ndim == 0:  # 0-d arrays: one entry
        return search_stop(data, int(highs), int(lows), freq, int(starts))
    if starts.ndim != 1:
        raise ValueError("start_idx, target_high and target_low must broadcast to one 1-D shape")
    starts, highs, lows = (np.ascontiguousarray(a) for a in (starts, highs, lows))
    not_start = data[starts, chain[0, 0]] == 0
    if not_start.any():
        raise ValueError(f"{not_start.sum()} start rows are not the first tick of a {freq} bar, e.g. row {starts[not_start][0]}")
    sides = _search_stop_many(data, starts, chain, PRICE_COL, highs, lows)
    broken = sides == BROKEN
    if broken.any():
        raise ValueError(f"bar summaries contradict the ticks for {broken.sum()} entries, e.g. the {freq} bar at row {starts[broken][0]}")
    return sides


@njit(parallel=True, cache=True)
def _search_stop_many(data, starts, chain, price_col, target_high, target_low):
    out = np.empty(len(starts), dtype=np.int8)
    for q in prange(len(starts)):
        out[q] = _search_stop(data, starts[q], chain, price_col, target_high[q], target_low[q])
    return out


def load_dat(path):
    """Memory-map tick.dat with the column layout to_dat.py wrote."""
    row_bytes = len(DAT_COLS) * 8
    size = os.path.getsize(path)
    if size % row_bytes:
        raise ValueError(f"{path}: {size} bytes is not a whole number of {len(DAT_COLS)}-column rows")
    return np.memmap(path, dtype=np.int64, mode='r', shape=(size // row_bytes, len(DAT_COLS)))
