import os
import sys

import numpy as np
from rich.console import Console

from stop_search import first_hit, first_hit_many, load_dat, PRICE_COL
from benchmarks.benchmark_search import make_queries

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
    memmap_path = 'data/dat_test/tick.dat'

    if not os.path.exists(memmap_path):
        console.print(f"[red]Could not find test data at {memmap_path}[/red]")
        return

    data = load_dat(memmap_path)
    # The same queries benchmark_search.py runs, so a mismatch it reports can be replayed here
    starts, ends, uppers, lowers = make_queries(data, freq)
    array_sides = first_hit_many(data, freq, starts, uppers, lowers)

    for q, (idx, end_idx, tp, sl) in enumerate(zip(starts.tolist(), ends.tolist(), uppers.tolist(), lowers.tolist())):
        hit_idx, bf_side = brute_force_search(data, idx, end_idx, tp, sl)
        single_side = first_hit(data, freq, idx, tp, sl)
        array_side = int(array_sides[q])

        if not bf_side == single_side == array_side:
            print("\n" + "="*50)
            print("MISMATCH FOUND!")
            print(f"Input Parameters:")
            print(f"  query: {q}")
            print(f"  freq: {freq}")
            print(f"  start_idx: {idx} (bar ends at {end_idx})")
            print(f"  upper: {tp}")
            print(f"  lower: {sl}")
            print(f"\nResults:")
            print(f"  Brute Force returned:         {bf_side} (tick {hit_idx})")
            print(f"  first_hit returned:           {single_side}")
            print(f"  first_hit_many returned:      {array_side}")

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

    console.print(f"[green]No mismatch in {len(starts):,} queries on {freq} bars (first_hit and first_hit_many).[/green]")


if __name__ == "__main__":
    main()
