import numpy as np
from numba import njit

import json
import zarr

def load_col_names_from_zarr(zarr_path='data/zarr/tick.zarr'):
    try:
        z = zarr.open(zarr_path, mode='r')
        return z.attrs['column_names']
    except Exception:
        # Fallback if building fresh without Zarr
        base = ['start_ind', 'ts', 'price']
        for f in ['day', '60', '30', '15', '10', '5', '1', '15s', '1s']:
            base.extend([f'high_{f}', f'low_{f}', f'next_ind_{f}'])
        return base

col_names = load_col_names_from_zarr()

col_to_ind_dict = {col_name: index for index, col_name in enumerate(col_names)}

# Largest bars first, so a clean day/hour is skipped in one step.
freqs = ['day', '60', '30', '15', '10', '5', '1', '15s', '1s']
JUMP_COLS = np.array([
    (col_to_ind_dict[f'next_ind_{f}'], col_to_ind_dict[f'high_{f}'], col_to_ind_dict[f'low_{f}'])
    for f in freqs
], dtype=np.int64)
PRICE_COL = col_to_ind_dict['price']


@njit(cache=True)
def _search(data, start_idx, target_high, target_low, jump_cols, price_col, side_only):
    n = data.shape[0]
    i = start_idx
    while i < n:
        price = data[i, price_col]
        if price >= target_high:
            return i, 1
        if price <= target_low:
            return i, -1
        # Bar summaries live only on a bar's first row, where they cover exactly [i, next_ind),
        # so nothing before the entry can leak in.
        jumped = False
        for k in range(jump_cols.shape[0]):
            nxt = data[i, jump_cols[k, 0]]
            if nxt == 0:
                continue
            hit_high = data[i, jump_cols[k, 1]] >= target_high
            hit_low = data[i, jump_cols[k, 2]] <= target_low
            if not hit_high and not hit_low:
                i = nxt
                jumped = True
                break
            if side_only and hit_high != hit_low:
                # Only one level is inside this bar and nothing before it hit: that side is first.
                return -1, 1 if hit_high else -1
        if not jumped:
            i += 1
    return -1, 0


def search(data: np.ndarray, start_idx: int, target_high: int, target_low: int, side_only: bool = False):
    """
    Finds the first tick from start_idx onwards where price >= target_high or price <= target_low.
    """

    idx, side = _search(data, start_idx, target_high, target_low, JUMP_COLS, PRICE_COL, side_only)
    if side == 0:
        return None, None
    return (None if idx < 0 else int(idx)), int(side)
