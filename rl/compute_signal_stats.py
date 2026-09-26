"""Compute per-family distribution stats for env-mask calibration.

Reads data/normalised_trading_data_np.npz (regenerate via data_pipeline/to_numpy.py
if missing) and prints percentile breakdowns for the 3 signals used by the env-side
safety mask (bar_pnl_raw, ewm_mean, ewm_sharpe), per family and per signal.

Usage:
    python training/compute_signal_stats.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import params

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
NPZ_PATH = os.path.join(DATA_DIR, 'normalised_trading_data_np.npz')

PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
SIGNAL_COLS = {'bar_pnl_raw': 0, 'ewm_mean': 3, 'ewm_sharpe': 4}


def family_pcts(arr_3d, sig_col):
    """arr_3d: (N, 72, 5). Returns (all_pcts, active_pcts, active_rate, n_total, n_active)."""
    sig = arr_3d[:, :, sig_col].ravel()
    side = arr_3d[:, :, 1].ravel()
    active = side != 0
    sig_active = sig[active]
    pcts_all = {p: float(np.percentile(sig, p)) for p in PCTS} if sig.size else {}
    pcts_act = {p: float(np.percentile(sig_active, p)) for p in PCTS} if sig_active.size else {}
    return pcts_all, pcts_act, float(active.mean()), int(sig.size), int(sig_active.size)


def main():
    if not os.path.exists(NPZ_PATH):
        print(f'NPZ not found: {NPZ_PATH}')
        print('Regenerate via: python data_pipeline/to_numpy.py')
        sys.exit(1)

    print(f'Loading {NPZ_PATH}')
    npz = np.load(NPZ_PATH, allow_pickle=False)
    print(f'Variant arrays found: {[k for k in npz.files if k.endswith("_variants")]}')

    out = {}
    for fam in params.STRAT_FAMILIES:
        key = f'{fam}_variants'
        if key not in npz.files:
            print(f'  {fam}: missing in npz, skip')
            continue
        arr = npz[key]
        out[fam] = {}
        print(f'\n=== {fam} ({arr.shape[0]:,} bars x 72 variants) ===')
        for sig_name, sig_col in SIGNAL_COLS.items():
            all_p, act_p, act_rate, n_total, n_active = family_pcts(arr, sig_col)
            out[fam][sig_name] = {
                'pcts_all': all_p, 'pcts_active': act_p,
                'active_rate': act_rate, 'n_total': n_total, 'n_active': n_active,
            }
            top = ' '.join(f'{p}%={act_p.get(p, 0):.4f}' for p in PCTS)
            print(f'  {sig_name:15s} active_rate={act_rate:.3f} pcts(active)={top}')

    out_path = os.path.join(os.path.dirname(__file__), 'signal_stats.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
