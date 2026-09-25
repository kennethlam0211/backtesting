import os
import sys
import time

import numpy as np
from rich.console import Console

from to_dat import DAT_COLS
from search_stop import search_stop, search_stop_batch, load_dat, PRICE_COL

console = Console()


def brute_force_search(data: np.ndarray, start_idx: int, end_idx: int, target_high: int, target_low: int):
    """
    Naive tick-by-tick scan of the same bar. Simulates a traditional backtester stepping through every row.
    """
    for i in range(start_idx, end_idx):
        price = data[i, PRICE_COL]
        if price >= target_high:
            return 1
        if price <= target_low:
            return -1
    return 0


def main():
    freq = sys.argv[1] if len(sys.argv) > 1 else '1'  # bar size the entries sit on, e.g. 1, 15, 60, day

    console.print("[cyan]Opening memory-mapped .dat array...[/cyan]")
    memmap_path = 'data/zarr_test/tick.dat'

    if not os.path.exists(memmap_path):
        console.print(f"[red]Could not find test data at {memmap_path}[/red]")
        return

    data = load_dat(memmap_path)
    console.print(f"[green]Array contains {data.shape[0]:,} ticks.[/green]")

    # Entries sit on the first tick of a bar; each search looks only inside that bar
    next_col = DAT_COLS.index(f'next_ind_{freq}')
    bar_starts = np.flatnonzero(data[:, next_col])

    np.random.seed(42)
    NUM_QUERIES = 10_000
    start_indices = np.random.choice(bar_starts, size=NUM_QUERIES)
    end_indices = data[start_indices, next_col]
    start_prices = data[start_indices, PRICE_COL]
    # Target TP and SL (e.g., +/- 10 to 50 points, remembering price is x100)
    target_highs = start_prices + np.random.randint(1000, 5000, size=NUM_QUERIES)
    target_lows = start_prices - np.random.randint(1000, 5000, size=NUM_QUERIES)

    console.print(f"\n[yellow]Running {NUM_QUERIES:,} Brute Force tick-by-tick searches on {freq} bars...[/yellow]")
    t0 = time.perf_counter()
    bf_results = [brute_force_search(data, s, e, h, l) for s, e, h, l in zip(start_indices, end_indices, target_highs, target_lows)]
    bf_time = time.perf_counter() - t0
    console.print(f"Brute Force took: [red]{bf_time:.4f} seconds[/red] ({(bf_time/NUM_QUERIES)*1000:.3f} ms per query)")

    # Warm up Numba JIT (run once so compilation time isn't counted in the benchmark)
    search_stop(data, int(target_highs[0]), int(target_lows[0]), freq, int(start_indices[0]))
    search_stop_batch(data, target_highs[:1], target_lows[:1], freq, start_indices[:1])

    console.print(f"\n[yellow]Running {NUM_QUERIES:,} search_stop calls...[/yellow]")
    t0 = time.perf_counter()
    single_results = [search_stop(data, int(h), int(l), freq, int(s)) for s, h, l in zip(start_indices, target_highs, target_lows)]
    single_time = time.perf_counter() - t0
    console.print(f"search_stop took: [green]{single_time:.4f} seconds[/green] ({(single_time/NUM_QUERIES)*1000:.3f} ms per query)")

    console.print(f"\n[yellow]Running {NUM_QUERIES:,} searches in one search_stop_batch call...[/yellow]")
    t0 = time.perf_counter()
    batch_results = search_stop_batch(data, target_highs, target_lows, freq, start_indices).tolist()
    batch_time = time.perf_counter() - t0
    console.print(f"search_stop_batch took: [green]{batch_time:.4f} seconds[/green] ({(batch_time/NUM_QUERIES)*1000:.4f} ms per query)")

    # Verify correctness
    mismatches = sum(b != s or b != p for b, s, p in zip(bf_results, single_results, batch_results))
    if mismatches == 0:
        console.print("\n[bold green]SUCCESS: All three returned identical results 100% of the time![/bold green]")
    else:
        console.print(f"\n[bold red]ERROR: {mismatches} results did not match! Run run_benchmark_mismatch.py {freq}[/bold red]")

    console.print(f"\n[bold cyan]search_stop is {bf_time / single_time:.1f}x and search_stop_batch {bf_time / batch_time:.1f}x FASTER than Brute Force![/bold cyan]")


if __name__ == "__main__":
    main()
