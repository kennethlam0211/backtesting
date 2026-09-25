import os
import sys
import time

import numpy as np
from rich.console import Console

from stop_search import StopSearch, DAT_COLS

PRICE_COL = DAT_COLS.index('price')

console = Console()


def brute_force_search(data: np.ndarray, start_idx: int, end_idx: int, upper: int, lower: int):
    """
    Naive tick-by-tick scan of the same bar. Simulates a traditional backtester stepping through every row.
    """
    for i in range(start_idx, end_idx):
        price = data[i, PRICE_COL]
        if price >= upper:
            return 1
        if price <= lower:
            return -1
    return 0


def make_queries(data, freq, num_queries=10_000, seed=42):
    """
    Random entries on the first tick of `freq` bars, each with an upper and a lower level 10 to 50 pts
    from the entry price (price is x4, in 0.25-pt ticks: 40 to 200 ticks). Shared with run_benchmark_mismatch.py so both scripts check
    exactly the same queries. Returns (starts, bar ends, uppers, lowers).
    """
    next_col = DAT_COLS.index(f'next_ind_{freq}')
    rng = np.random.default_rng(seed)
    starts = rng.choice(np.flatnonzero(data[:, next_col]), size=num_queries)
    ends = data[starts, next_col]
    prices = data[starts, PRICE_COL]
    return starts, ends, prices + rng.integers(40, 200, num_queries), prices - rng.integers(40, 200, num_queries)


def main():
    freq = sys.argv[1] if len(sys.argv) > 1 else '1'  # bar size the entries sit on, e.g. 1, 15, 60, day

    console.print("[cyan]Opening memory-mapped .dat array...[/cyan]")
    memmap_path = 'data/processed/tick.dat'

    if not os.path.exists(memmap_path):
        console.print(f"[red]Could not find test data at {memmap_path}[/red]")
        return

    stops = StopSearch.load(memmap_path)
    data = stops.data
    console.print(f"[green]Array contains {data.shape[0]:,} ticks.[/green]")

    NUM_QUERIES = 10_000
    start_indices, end_indices, uppers, lowers = make_queries(data, freq, NUM_QUERIES)

    console.print(f"\n[yellow]Running {NUM_QUERIES:,} Brute Force tick-by-tick searches on {freq} bars...[/yellow]")
    t0 = time.perf_counter()
    bf_results = [brute_force_search(data, s, e, h, l) for s, e, h, l in zip(start_indices, end_indices, uppers, lowers)]
    bf_time = time.perf_counter() - t0
    console.print(f"Brute Force took: [red]{bf_time:.4f} seconds[/red] ({(bf_time/NUM_QUERIES)*1000:.3f} ms per query)")

    # Warm up Numba JIT (run once so compilation time isn't counted in the benchmark)
    stops.first_hit(freq, int(start_indices[0]), int(uppers[0]), int(lowers[0]))
    stops.first_hit_many(freq, start_indices[:1], uppers[:1], lowers[:1])

    console.print(f"\n[yellow]Running {NUM_QUERIES:,} first_hit calls, one entry each...[/yellow]")
    t0 = time.perf_counter()
    single_results = [stops.first_hit(freq, int(s), int(h), int(l)) for s, h, l in zip(start_indices, uppers, lowers)]
    single_time = time.perf_counter() - t0
    console.print(f"first_hit took: [green]{single_time:.4f} seconds[/green] ({(single_time/NUM_QUERIES)*1000:.3f} ms per query)")

    console.print(f"\n[yellow]Running {NUM_QUERIES:,} searches in one first_hit_many call...[/yellow]")
    t0 = time.perf_counter()
    batch_results = stops.first_hit_many(freq, start_indices, uppers, lowers).tolist()
    batch_time = time.perf_counter() - t0
    console.print(f"first_hit_many took: [green]{batch_time:.4f} seconds[/green] ({(batch_time/NUM_QUERIES)*1000:.4f} ms per query)")

    # Verify correctness
    mismatches = sum(b != s or b != p for b, s, p in zip(bf_results, single_results, batch_results))
    if mismatches == 0:
        console.print("\n[bold green]SUCCESS: All three returned identical results 100% of the time![/bold green]")
    else:
        console.print(f"\n[bold red]ERROR: {mismatches} results did not match! Run: python -m benchmarks.run_benchmark_mismatch {freq}[/bold red]")

    console.print(f"\n[bold cyan]first_hit is {bf_time / single_time:.1f}x and first_hit_many {bf_time / batch_time:.1f}x FASTER than Brute Force![/bold cyan]")


if __name__ == "__main__":
    main()
