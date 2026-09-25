import os

import numpy as np
from numba import njit, prange

from .params import CHILD, DAT_COLS


def _seconds(freq):
    if freq == 'day':
        return 24 * 3600
    return int(freq[:-1]) if freq.endswith('s') else int(freq) * 60


def _check_child(child):
    for parent, c in child.items():
        if c is None:
            continue
        if c not in child:
            raise ValueError(f"CHILD maps {parent} to {c}, which has no entry of its own")
        if _seconds(c) >= _seconds(parent) or _seconds(parent) % _seconds(c):
            raise ValueError(f"{c} bars do not tile {parent} bars (a child must be smaller and divide its parent)")


def _chains(cols, child):
    """Per bar size: (next_ind, high, low) columns from that size down to its last child."""
    ix = {c: k for k, c in enumerate(cols)}
    chains = {}
    for freq in child:
        rows, f = [], freq
        while f is not None:
            rows.append((ix[f'next_ind_{f}'], ix[f'high_{f}'], ix[f'low_{f}']))
            f = child[f]
        chains[freq] = np.array(rows, dtype=np.int64)
    return chains


_check_child(CHILD)
PRICE_COL = DAT_COLS.index('price')
CHAINS = _chains(DAT_COLS, CHILD)

_INT = (int, np.integer)

# Kernel result when the bar summaries contradict the ticks. Returned rather than raised:
# an exception inside a prange loop is dropped silently.
BROKEN = 2


@njit(cache=True)
def _kernel_one(data, ind, chain, price_col, upper, lower):
    n = data.shape[0]
    depth = chain.shape[0]
    ends = np.empty(depth, dtype=np.int64)  # ends[k]: end of the parent bar whose children level k scans
    ends[0] = data[ind, chain[0, 0]]
    k = 0
    while True:
        if ind >= ends[k]:
            return BROKEN  # a bar held a level but none of its sub-bars did
        nxt = data[ind, chain[k, 0]]
        if nxt <= ind or nxt > n:
            return BROKEN  # sub-bar missing, or a pointer outside the data (sliced or cut array)
        hit_upper = data[ind, chain[k, 1]] >= upper
        hit_lower = data[ind, chain[k, 2]] <= lower
        if hit_upper != hit_lower:
            return 1 if hit_upper else -1
        if not hit_upper:
            if k == 0:
                return 0  # nothing inside the entry bar: skip
            ind = nxt  # next bar of the same size, still inside the parent
        elif k + 1 < depth:
            k += 1  # both inside: split into the child bars, which start on the same row
            ends[k] = nxt
        else:
            for t in range(ind, nxt):  # both inside a 1s bar: walk its ticks
                price = data[t, price_col]
                if price >= upper:
                    return 1
                if price <= lower:
                    return -1
            return BROKEN  # a 1s bar held a level but none of its ticks did


@njit(parallel=True, cache=True)
def _kernel_many(data, starts, chain, price_col, upper, lower):
    out = np.empty(len(starts), dtype=np.int8)
    for q in prange(len(starts)):
        out[q] = _kernel_one(data, starts[q], chain, price_col, upper[q], lower[q])
    return out


# Float levels are clipped here (inf means "no level"); far beyond any price in ticks
_LEVEL_LIMIT = 2 ** 62


def _as_levels(upper, lower):
    """Integer levels with the same hits: prices are integers, so price >= u <=> price >= ceil(u)
    and price <= l <=> price <= floor(l)."""
    upper, lower = np.asarray(upper), np.asarray(lower)
    if upper.dtype.kind == 'f' or lower.dtype.kind == 'f':
        if np.isnan(upper).any() or np.isnan(lower).any():
            raise ValueError("upper and lower must not be NaN")
        upper = np.clip(np.ceil(upper), -_LEVEL_LIMIT, _LEVEL_LIMIT)
        lower = np.clip(np.floor(lower), -_LEVEL_LIMIT, _LEVEL_LIMIT)
    return upper.astype(np.int64, copy=False), lower.astype(np.int64, copy=False)


def _one(data, chain, price_col, freq, start_idx, upper, lower):
    if not (isinstance(start_idx, _INT) and isinstance(upper, _INT) and isinstance(lower, _INT)):
        if np.ndim(start_idx) or np.ndim(upper) or np.ndim(lower):
            raise TypeError("first_hit takes one entry; use first_hit_many for arrays")
        if np.asarray(start_idx).dtype.kind not in 'iu':
            raise TypeError("start_idx must be an integer row number")
        upper, lower = (int(x) for x in _as_levels(upper, lower))  # floats, 0-d arrays
    start_idx = int(start_idx)
    if not 0 <= start_idx < data.shape[0] or data[start_idx, chain[0, 0]] == 0:
        raise ValueError(f"row {start_idx} is not the first tick of a {freq} bar")
    side = int(_kernel_one(data, start_idx, chain, price_col, int(upper), int(lower)))
    if side == BROKEN:
        raise ValueError(f"the {freq} bar at row {start_idx} is broken: its summary contradicts its ticks or points outside the data")
    return side


def _many(data, chain, price_col, freq, start_idx, upper, lower):
    starts = np.asarray(start_idx)
    if starts.ndim != 1:
        raise TypeError("first_hit_many takes a 1-D array of start rows; use first_hit for one entry")
    if starts.size and starts.dtype.kind not in 'iu':
        raise TypeError("start_idx must be integer row numbers")
    starts = starts.astype(np.int64, copy=False)
    uppers, lowers = _as_levels(upper, lower)
    try:
        shape = np.broadcast_shapes(starts.shape, uppers.shape, lowers.shape)
    except ValueError:
        shape = None
    if shape != starts.shape:
        raise ValueError("upper and lower must be single values or arrays as long as start_idx")
    # Single levels are expanded into real arrays: the kernel gets contiguous memory, never broadcast views
    starts, uppers, lowers = (np.ascontiguousarray(a) if a.shape == shape else np.broadcast_to(a, shape).copy()
                              for a in (starts, uppers, lowers))
    outside = (starts < 0) | (starts >= data.shape[0])
    if outside.any():
        raise ValueError(f"{outside.sum()} start rows are outside the data, e.g. row {starts[outside][0]}")
    not_start = data[starts, chain[0, 0]] == 0
    if not_start.any():
        raise ValueError(f"{not_start.sum()} start rows are not the first tick of a {freq} bar, e.g. row {starts[not_start][0]}")
    sides = _kernel_many(data, starts, chain, price_col, uppers, lowers)
    broken = sides == BROKEN
    if broken.any():
        raise ValueError(f"{broken.sum()} {freq} bars are broken (summary contradicts the ticks or points outside the data), e.g. row {starts[broken][0]}")
    return sides


def first_hit(data, freq, start_idx, upper, lower):
    """
    Which level one entry's `freq` bar touches first, looking only inside that bar.
    For many entries at once use first_hit_many; for use from another class, TickData holds the
    loaded ticks and offers both calls.

    Same logic as search_org.search_stop: one level inside a bar decides it; both inside splits the bar
    into its CHILD bars, checked in time order; a 1s bar with both inside is walked tick by tick.

    Args:
        data: the whole tick.dat as an int64 array (see load_dat). next_ind values are row numbers in
            the full file, so a slice must start at row 0 (a cut end is detected and raises).
        freq: bar size to look inside: 'day', '60', '30', '15', '10', '5', '1', '15s' or '1s'.
        start_idx: entry row; must be the first tick of a `freq` bar.
        upper: upper level in ticks, price x4 (4000.25 -> 16001); hit when price >= upper.
        lower: lower level in ticks; hit when price <= lower.
            Float levels are exact (upper is rounded up, lower down, to whole ticks);
            inf / -inf means no upper / lower level; NaN raises.

    Returns:
         1  upper is hit first (a long's take-profit, a short's stop)
        -1  lower is hit first (a long's stop, a short's take-profit)
         0  neither is hit inside the bar (skip)

    Raises:
        TypeError: an argument is an array (use first_hit_many).
        ValueError: start_idx is outside the data or not the first tick of a `freq` bar, a level is
            NaN, or the bar is broken (its summary contradicts its ticks or points outside the data).

    Example:
        import numpy as np
        from tick_data import first_hit, load_dat, DAT_COLS, PRICE_COL

        data = load_dat('data/zarr/tick.dat')
        i = np.flatnonzero(data[:, DAT_COLS.index('next_ind_1')])[0]    # first tick of the first 1-min bar
        entry = data[i, PRICE_COL]
        side = first_hit(data, '1', i, entry + 40, entry - 20)         # +10 pts / -5 pts (40 / 20 ticks)
        # side == 1: +10 came first; -1: -5 came first; 0: neither within that minute
    """
    return _one(data, CHAINS[freq], PRICE_COL, freq, start_idx, upper, lower)


def first_hit_many(data, freq, start_idx, upper, lower):
    """
    first_hit for many entries at once, all on `freq` bars: entry q checks the bar starting at
    start_idx[q] against upper[q] and lower[q], and the results equal calling first_hit per entry.
    Runs in one compiled call across all CPU cores (NUMBA_NUM_THREADS caps it).

    Args:
        data: the whole tick.dat as an int64 array (see load_dat).
        freq: bar size shared by all entries: 'day', '60', '30', '15', '10', '5', '1', '15s' or '1s'.
        start_idx: 1-D array of entry rows, each the first tick of a `freq` bar.
        upper, lower: arrays as long as start_idx, or single values used for every entry; in ticks,
            same rules as first_hit (floats exact, inf = no level, NaN raises).

    Returns:
        int8 array with one value per entry, in input order: 1 upper first, -1 lower first, 0 neither
        inside the bar. Cast with .astype(np.int64) before doing arithmetic on it (int8 holds only
        -128..127); .tolist() gives a plain list.

    Raises:
        TypeError: start_idx is not a 1-D array (use first_hit for one entry).
        ValueError: upper/lower do not match start_idx in length, a start row is outside the data or
            not the first tick of a `freq` bar, a level is NaN, or a bar is broken.

    Example:
        import numpy as np
        from tick_data import first_hit_many, load_dat, DAT_COLS, PRICE_COL

        data = load_dat('data/zarr/tick.dat')
        starts = np.flatnonzero(data[:, DAT_COLS.index('next_ind_1')])  # first tick of every 1-min bar
        entry = data[starts, PRICE_COL]
        sides = first_hit_many(data, '1', starts, entry + 40, entry - 20)  # +10 / -5 pts each
        print((sides == 1).mean(), (sides == -1).mean(), (sides == 0).mean())  # share of each outcome
    """
    return _many(data, CHAINS[freq], PRICE_COL, freq, start_idx, upper, lower)


def load_dat(path, n_cols=len(DAT_COLS)):
    """Memory-map tick.dat read-only; n_cols is the column count pipeline/to_dat.py wrote (len(DAT_COLS))."""
    row_bytes = n_cols * 8
    size = os.path.getsize(path)
    if size % row_bytes:
        raise ValueError(f"{path}: {size} bytes is not a whole number of {n_cols}-column rows")
    return np.memmap(path, dtype=np.int64, mode='r', shape=(size // row_bytes, n_cols))


def _whole_file(data, n_cols):
    """The file behind `data` if it is an entire tick.dat memmap (so it can be reopened), else None."""
    if (isinstance(data, np.memmap) and data.filename and data.offset == 0 and data.ndim == 2
            and data.shape[1] == n_cols and data.dtype == np.int64 and data.flags.c_contiguous
            and data.nbytes == os.path.getsize(data.filename)):
        return data.filename
    return None


def _stamp(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


class TickData:
    """
    tick.dat loaded once, with its bar layout, for another class (backtester, RL env) to hold and query.

    The ticks are memory-mapped read-only: loading is instant, pages are read from disk on first touch
    and stay in the OS page cache, shared by every process that opens the same file. Pickling (e.g.
    handing the owner to worker processes) carries only the absolute path, so each worker reopens the
    file instead of receiving a copy of the ticks; unpickling raises if the file changed meanwhile. Workers that run in parallel should set
    NUMBA_NUM_THREADS=1 and be started with the 'spawn' or 'forkserver' method (numba's OpenMP
    threads do not survive fork).

    Example:
        from tick_data import TickData

        class Backtester:
            def __init__(self, path):
                self.ticks = TickData.load(path)                  # once

            def label(self, freq, tp, sl):
                starts = self.ticks.bar_starts(freq)              # first tick of every `freq` bar (cached)
                entry = self.ticks.price(starts)
                return self.ticks.first_hit_many(freq, starts, entry + tp, entry - sl)  # 1 / -1 / 0 each

        bt = Backtester('data/zarr/tick.dat')
        sides = bt.label('1', 40, 20)                             # +10 / -5 pts (ticks) on every 1-min bar
    """

    def __init__(self, data, path=None, cols=DAT_COLS, child=CHILD):
        _check_child(child)
        if path is None:
            path = _whole_file(data, len(cols))
        self.data = data
        self.path = None if path is None else os.path.abspath(os.fspath(path))
        self._stamp = None if self.path is None else _stamp(self.path)
        self.cols = list(cols)
        self.child = dict(child)
        self.price_col = self.cols.index('price')
        self._chains = _chains(self.cols, self.child)
        self._starts = {}

    @classmethod
    def load(cls, path, cols=DAT_COLS, child=CHILD):
        """Memory-map the tick.dat at `path` (see load_dat)."""
        return cls(load_dat(path, len(cols)), path=path, cols=cols, child=child)

    def __len__(self):
        return self.data.shape[0]

    def price(self, idx):
        """Price in ticks (x4) at row(s) idx."""
        return self.data[idx, self.price_col]

    def bar_starts(self, freq):
        """Rows that start a `freq` bar, ascending, read-only. The column is scanned once, then cached."""
        if freq not in self._starts:
            starts = np.flatnonzero(self.data[:, self._chains[freq][0, 0]])
            starts.setflags(write=False)  # shared cache: callers must copy before changing it
            self._starts[freq] = starts
        return self._starts[freq]

    def bar_end(self, freq, start_idx):
        """End row (exclusive) of the `freq` bar(s) starting at start_idx."""
        return self.data[start_idx, self._chains[freq][0, 0]]

    def first_hit(self, freq, start_idx, upper, lower):
        """first_hit on these ticks: one entry -> 1 upper first, -1 lower first, 0 neither inside the bar."""
        return _one(self.data, self._chains[freq], self.price_col, freq, start_idx, upper, lower)

    def first_hit_many(self, freq, start_idx, upper, lower):
        """first_hit_many on these ticks: many entries -> int8 array of 1 / -1 / 0, in input order."""
        return _many(self.data, self._chains[freq], self.price_col, freq, start_idx, upper, lower)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_starts'] = {}
        if self.path is not None:
            state['data'] = None  # reopened from path when unpickled, not copied
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.data is None:
            if _stamp(self.path) != self._stamp:
                raise ValueError(f"{self.path} changed since it was loaded; load it again instead of unpickling")
            self.data = load_dat(self.path, len(self.cols))
