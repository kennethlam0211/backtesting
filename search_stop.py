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
    Which level the `freq` bar starting at start_idx touches first, looking only inside that bar.
    Returns 1 if target_high (price >= target_high) comes first, -1 if target_low (price <= target_low)
    does, 0 if neither is touched (skip).

    data is tick.dat as an int64 array (see load_dat); start_idx must be the bar's first tick.
    Same logic as search_org.search_stop: one level inside a bar decides it; both inside splits the bar
    into its CHILD bars, checked in time order; a 1s bar with both inside is walked tick by tick.
    """
    chain = CHAINS[freq]
    if data[start_idx, chain[0, 0]] == 0:
        raise ValueError(f"row {start_idx} is not the first tick of a {freq} bar")
    side = int(_search_stop(data, start_idx, chain, PRICE_COL, target_high, target_low))
    if side == BROKEN:
        raise ValueError(f"bar summaries of the {freq} bar at row {start_idx} contradict its ticks")
    return side


@njit(parallel=True, cache=True)
def _search_stop_batch(data, starts, chain, price_col, target_high, target_low):
    out = np.empty(len(starts), dtype=np.int8)
    for q in prange(len(starts)):
        out[q] = _search_stop(data, starts[q], chain, price_col, target_high[q], target_low[q])
    return out


def search_stop_batch(data, target_high, target_low, freq, start_idx):
    """
    search_stop for many entries at once: equal-length arrays of levels and start rows, all on `freq`
    bars. Runs in one compiled call across all cores (NUMBA_NUM_THREADS caps it); returns int8 sides.
    """
    start_idx = np.asarray(start_idx, dtype=np.int64)
    target_high = np.asarray(target_high, dtype=np.int64)
    target_low = np.asarray(target_low, dtype=np.int64)
    if not (start_idx.ndim == 1 and start_idx.shape == target_high.shape == target_low.shape):
        raise ValueError("start_idx, target_high and target_low must be 1-D arrays of equal length")
    chain = CHAINS[freq]
    not_start = data[start_idx, chain[0, 0]] == 0
    if not_start.any():
        raise ValueError(f"{not_start.sum()} start rows are not the first tick of a {freq} bar, e.g. row {start_idx[not_start][0]}")
    sides = _search_stop_batch(data, start_idx, chain, PRICE_COL, target_high, target_low)
    broken = sides == BROKEN
    if broken.any():
        raise ValueError(f"bar summaries contradict the ticks for {broken.sum()} entries, e.g. the {freq} bar at row {start_idx[broken][0]}")
    return sides


def load_dat(path):
    """Memory-map tick.dat with the column layout to_dat.py wrote."""
    row_bytes = len(DAT_COLS) * 8
    size = os.path.getsize(path)
    if size % row_bytes:
        raise ValueError(f"{path}: {size} bytes is not a whole number of {len(DAT_COLS)}-column rows")
    return np.memmap(path, dtype=np.int64, mode='r', shape=(size // row_bytes, len(DAT_COLS)))
