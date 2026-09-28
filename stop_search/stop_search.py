import os

import numpy as np
from numba import njit, prange

from .params import CHILD, DAT_DTYPE, DAT_PATH, INDEX_COLS, PRICE_COLS, ROW_BYTES


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


def _chains(child):
    """Per bar size: its next_ind column (in the uint32 view) and high / low columns (in the uint16 view), from that
    size down to its last child."""
    ix, px = {c: k for k, c in enumerate(INDEX_COLS)}, {c: k for k, c in enumerate(PRICE_COLS)}
    chains = {}
    for freq in child:
        rows, f = [], freq
        while f is not None:
            rows.append((ix[f'next_ind_{f}'], px[f'high_{f}'], px[f'low_{f}']))
            f = child[f]
        chains[freq] = np.array(rows, dtype=np.int64)
    return chains


def _views(data):
    """tick.dat rows (a DAT_DTYPE array or memmap) as its two tables: the uint32 columns (INDEX_COLS) and the uint16
    ones (PRICE_COLS), each rows x columns, sharing the rows' memory."""
    if data.dtype != DAT_DTYPE or data.ndim != 1 or not data.flags.c_contiguous:
        raise ValueError('tick.dat data must be a contiguous 1-D array of stop_search.params.DAT_DTYPE rows')
    raw = data.view(np.uint8).reshape(len(data), ROW_BYTES)
    n_ix = len(INDEX_COLS)
    return raw[:, :4 * n_ix].view('<u4'), raw[:, 4 * n_ix:4 * n_ix + 2 * len(PRICE_COLS)].view('<u2')


def _chain_for(chains, freq):
    try:
        return chains[freq]
    except (KeyError, TypeError):
        raise ValueError(f"unknown bar size {freq!r}; use one of {', '.join(repr(f) for f in chains)}") from None


_INT = (int, np.integer)

# Kernel result when the bar summaries contradict the ticks. Returned rather than raised:
# an exception inside a prange loop is dropped silently.
BROKEN = 2


@njit(cache=True)
def _kernel_one(ix, px, ind, chain, price_col, upper, lower):
    # ix: the uint32 columns (next_ind), px: the uint16 ones (prices); both read as int64, never subtracted unsigned
    n = ix.shape[0]
    depth = chain.shape[0]
    ends = np.empty(depth, dtype=np.int64)  # ends[k]: end of the parent bar whose children level k scans
    ends[0] = np.int64(ix[ind, chain[0, 0]])
    k = 0
    while True:
        if ind >= ends[k]:
            return BROKEN  # a bar held a level but none of its sub-bars did
        nxt = np.int64(ix[ind, chain[k, 0]])
        if nxt <= ind or nxt > n:
            return BROKEN  # sub-bar missing, or a pointer outside the data (sliced or cut array)
        hit_upper = np.int64(px[ind, chain[k, 1]]) >= upper
        hit_lower = np.int64(px[ind, chain[k, 2]]) <= lower
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
                price = np.int64(px[t, price_col])
                if price >= upper:
                    return 1
                if price <= lower:
                    return -1
            return BROKEN  # a 1s bar held a level but none of its ticks did


@njit(parallel=True, cache=True)
def _kernel_many(ix, px, starts, chain, price_col, upper, lower):
    out = np.empty(len(starts), dtype=np.int8)
    for q in prange(len(starts)):
        out[q] = _kernel_one(ix, px, starts[q], chain, price_col, upper[q], lower[q])
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


def _one(ix, px, chain, price_col, freq, start_idx, upper, lower):
    if not (isinstance(start_idx, _INT) and isinstance(upper, _INT) and isinstance(lower, _INT)):
        if np.ndim(start_idx) or np.ndim(upper) or np.ndim(lower):
            raise TypeError("first_hit takes one entry; use first_hit_many for arrays")
        if np.asarray(start_idx).dtype.kind not in 'iu':
            raise TypeError("start_idx must be an integer row number")
        upper, lower = (int(x) for x in _as_levels(upper, lower))  # floats, 0-d arrays
    start_idx = int(start_idx)
    if not 0 <= start_idx < ix.shape[0] or ix[start_idx, chain[0, 0]] == 0:
        raise ValueError(f"row {start_idx} is not the first tick of a {freq} bar")
    side = int(_kernel_one(ix, px, start_idx, chain, price_col, int(upper), int(lower)))
    if side == BROKEN:
        raise ValueError(f"the {freq} bar at row {start_idx} is broken: its summary contradicts its ticks or points outside the data")
    return side


def _many(ix, px, chain, price_col, freq, start_idx, upper, lower):
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
    outside = (starts < 0) | (starts >= ix.shape[0])
    if outside.any():
        raise ValueError(f"{outside.sum()} start rows are outside the data, e.g. row {starts[outside][0]}")
    not_start = ix[starts, chain[0, 0]] == 0
    if not_start.any():
        raise ValueError(f"{not_start.sum()} start rows are not the first tick of a {freq} bar, e.g. row {starts[not_start][0]}")
    sides = _kernel_many(ix, px, starts, chain, price_col, uppers, lowers)
    broken = sides == BROKEN
    if broken.any():
        raise ValueError(f"{broken.sum()} {freq} bars are broken (summary contradicts the ticks or points outside the data), e.g. row {starts[broken][0]}")
    return sides


def _load_dat(path):
    """Memory-map tick.dat read-only: rows of DAT_DTYPE, as data_pipeline/to_dat.py writes them."""
    size = os.path.getsize(path)
    if size % ROW_BYTES:
        raise ValueError(f"{path}: {size} bytes is not a whole number of {ROW_BYTES}-byte rows (an int64 tick.dat "
                         f"from before 2026-09-28? rebuild it with python -m data_pipeline.to_dat)")
    return np.memmap(path, dtype=DAT_DTYPE, mode='r', shape=(size // ROW_BYTES,))


class StopSearch:
    """
    Stop search over tick.dat: which level, upper or lower, an entry's bar touches first.
    Load it once and keep it on the class that uses it (the backtester).

        stops = StopSearch.load()        # params.DAT_PATH (to_dat.py's default output), or pass a path

        stops.first_hit(freq, start_idx, upper, lower)       one entry    -> 1, -1 or 0
        stops.first_hit_many(freq, starts, uppers, lowers)   many entries -> int8 array of 1 / -1 / 0
        stops.bar_end(freq, start_idx)                        row just after the bar's last tick
        stops.price(idx)                                      price in ticks at row(s)

    Prices and levels are in ticks (price x4): 10 points = 40. 1 = upper hit first (a long's
    take-profit, a short's stop), -1 = lower hit first (a long's stop, a short's take-profit),
    0 = neither inside the bar.

    Same logic as reference/search_org.py (search_stop): one level inside a bar decides it; both inside
    splits the bar into its CHILD bars, checked in time order; a 1s bar with both inside is walked tick by tick.

    The ticks are memory-mapped read-only: loading is instant, pages are read from disk on first touch
    and stay in the OS page cache, shared by every process that opens the same file. Entry rows are the first
    ticks of bars, e.g. the training data's start_ind (backtesting.backtest.read_training_data).

    Example:
        from stop_search import StopSearch, DAT_PATH

        class Backtester:
            def __init__(self, path):
                self.stops = StopSearch.load(path)                 # once

            def label(self, freq, starts, tp, sl):
                entry = self.stops.price(starts)                    # starts: first ticks of `freq` bars
                return self.stops.first_hit_many(freq, starts, entry + tp, entry - sl)  # 1 / -1 / 0 each

        bt = Backtester(DAT_PATH)
        df = read_training_data()                                   # one row per 1-min bar
        sides = bt.label('1', df['start_ind'].to_numpy(), 40, 20)   # +10 / -5 pts on every 1-min bar
    """

    def __init__(self, data, child=CHILD):
        """
        data: the whole tick.dat as a 1-D array of stop_search.params.DAT_DTYPE rows (StopSearch.load maps the file).
        next_ind values are row numbers in the full file, so an array cut from it must start at row 0 (a cut end is
        detected and raises).
        """
        _check_child(child)
        self.data = data
        self._ix, self._px = _views(data)
        self.child = dict(child)
        self.price_col = PRICE_COLS.index('price')
        self._chains = _chains(self.child)

    @classmethod
    def load(cls, path=DAT_PATH, child=CHILD):
        """Memory-map the tick.dat at `path` read-only, by default params.DAT_PATH (to_dat.py's default output)."""
        return cls(_load_dat(path), child=child)

    def price(self, idx):
        """Price in ticks (x4) at row(s) idx, as int64 (stored as uint16: unsigned subtraction would wrap)."""
        return self._px[idx, self.price_col].astype(np.int64)

    def bar_end(self, freq, start_idx):
        """End row (exclusive) of the `freq` bar(s) starting at start_idx, as int64."""
        return self._ix[start_idx, _chain_for(self._chains, freq)[0, 0]].astype(np.int64)

    def first_hit(self, freq, start_idx, upper, lower):
        """
        One entry: which level the `freq` bar starting at start_idx touches first.

        Args:
            freq: bar size: 'day', '60', '30', '15', '10', '5', '1', '15s', '5s' or '1s'.
            start_idx: one row (an int), the first tick of a `freq` bar (e.g. the training data's start_ind).
            upper: one level in ticks (price x4); hit when price >= upper.
            lower: one level in ticks; hit when price <= lower.
                Floats are exact (upper rounded up, lower down); inf / -inf = no level; NaN raises.

        Returns:
            1 upper hit first, -1 lower hit first, 0 neither inside the bar (an int).

        Raises:
            TypeError: an argument is an array; use first_hit_many.
            ValueError: start_idx is not the first tick of a `freq` bar or is outside the data, a level
                is NaN, or the bar is broken.

        Example:
            stops = StopSearch.load()
            i = int(read_training_data()['start_ind'][0])               # an entry row: a 1-min bar's first tick
            entry = stops.price(i)
            side = stops.first_hit('1', i, entry + 40, entry - 20)      # +10 / -5 pts -> 1, -1 or 0
        """
        return _one(self._ix, self._px, _chain_for(self._chains, freq), self.price_col, freq, start_idx, upper, lower)

    def first_hit_many(self, freq, start_idx, upper, lower):
        """
        Many entries on `freq` bars in one parallel call; the same answers as first_hit per entry.
        Runs in one compiled call across all CPU cores (NUMBA_NUM_THREADS caps it).

        The arguments are columns, not rows: entry q is (start_idx[q], upper[q], lower[q]). A list of
        (freq, start, upper, lower) tuples is not accepted; unpack it first (one freq per call):
            freqs, starts, uppers, lowers = zip(*entries)
            sides = stops.first_hit_many(freqs[0], starts, uppers, lowers)
        For entries on several bar sizes, call once per size.

        Args:
            freq: bar size shared by every entry: 'day', '60', '30', '15', '10', '5', '1', '15s', '5s' or '1s'.
            start_idx: 1-D array (or list) of entry rows, each the first tick of a `freq` bar.
            upper, lower: arrays as long as start_idx, or one value used for every entry; in ticks,
                same rules as first_hit.

        Returns:
            int8 array, one value per entry in input order: 1 upper first, -1 lower first, 0 neither.
            Cast with .astype(np.int64) before arithmetic on it; .tolist() gives a plain list.

        Raises:
            TypeError: start_idx is a single row; use first_hit.
            ValueError: upper / lower do not match start_idx in length, a start row is not the first
                tick of a `freq` bar or is outside the data, a level is NaN, or a bar is broken.

        Example:
            stops = StopSearch.load()
            starts = read_training_data()['start_ind'].to_numpy()       # every 1-min bar
            entry = stops.price(starts)
            sides = stops.first_hit_many('1', starts, entry + 40, entry - 20)   # +10 / -5 pts each
        """
        return _many(self._ix, self._px, _chain_for(self._chains, freq), self.price_col, freq, start_idx, upper,
                     lower)
