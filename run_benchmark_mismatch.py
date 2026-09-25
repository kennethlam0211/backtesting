import os
import sys

import numpy as np
from rich.console import Console

from tick_data import first_hit, load_dat, DAT_COLS, PRICE_COL

console = Console()


def brute_force_search(data: np.ndarray, start_idx: int, end_idx: int, upper: int, lower: int):
    for i in range(start_idx, end_idx):
        price = data[i, PRICE_COL]
        if price >= upper:
            return i, 1
        if price <= lower:
            return i, -1
    return None, 0


def main():
    freq = sys.argv[1] if len(sys.argv) > 1 else '1'
    console.print("[cyan]Opening memory-mapped .dat array...[/cyan]")
    memmap_path = 'data/zarr_test/tick.dat'

    if not os.path.exists(memmap_path):
        console.print(f"[red]Could not find test data at {memmap_path}[/red]")
        return

    data = load_dat(memmap_path)
    next_col = DAT_COLS.index(f'next_ind_{freq}')
    bar_starts = np.flatnonzero(data[:, next_col])

    np.random.seed(42)
    NUM_QUERIES = 10_000
    start_indices = np.random.choice(bar_starts, size=NUM_QUERIES)

    queries = []
    for idx in start_indices:
        start_price = data[idx, PRICE_COL]
        tp_offset = np.random.randint(1000, 5000)
        sl_offset = np.random.randint(1000, 5000)
        queries.append((int(idx), int(start_price + tp_offset), int(start_price - sl_offset)))

    for idx, tp, sl in queries:
        end_idx = int(data[idx, next_col])
        hit_idx, bf_side = brute_force_search(data, idx, end_idx, tp, sl)
        stop_side = first_hit(data, freq, idx, tp, sl)

        if bf_side != stop_side:
            print("\n" + "="*50)
            print("MISMATCH FOUND!")
            print(f"Input Parameters:")
            print(f"  freq: {freq}")
            print(f"  start_idx: {idx} (bar ends at {end_idx})")
            print(f"  upper: {tp}")
            print(f"  lower: {sl}")
            print(f"\nResults:")
            print(f"  Brute Force returned: {bf_side} (tick {hit_idx})")
            print(f"  first_hit returned: {stop_side}")

            last = min(end_idx, (hit_idx if hit_idx is not None else end_idx) + 2, idx + 200)
            print("\nData Subset (start_idx up to the first hit, at most 200 ticks):")
            print("Index | Price")
            print("-" * 20)
            for j in range(idx, last):
                marker = " <-- START" if j == idx else ""
                if j == hit_idx:
                    marker += " <-- BF FIRST HIT"
                print(f"{j:<5} | {data[j, PRICE_COL]}{marker}")

            print("="*50 + "\n")
            return

    console.print(f"[green]No mismatch in {NUM_QUERIES:,} queries on {freq} bars.[/green]")


if __name__ == "__main__":
    main()
