import time
import zarr
import numpy as np
from rich.console import Console

# Import the pre-compiled numba search function and columns
from search import search, col_to_ind_dict, PRICE_COL

console = Console()

def brute_force_search(data: np.ndarray, start_idx: int, target_high: int, target_low: int):
    """
    Naive tick-by-tick search. Simulates a traditional backtester stepping through every row.
    """
    n = data.shape[0]
    for i in range(start_idx, n):
        price = data[i, PRICE_COL]
        if price >= target_high:
            return i, 1
        if price <= target_low:
            return i, -1
    return None, None

def main():
    # We'll use a memory-mapped NumPy array instead of loading 2.7GB into RAM at once.
    # To do this safely, we will extract the Zarr array directly to a binary memmap file.
    import os

    console.print("[cyan]Opening memory-mapped .dat array...[/cyan]")
    memmap_path = 'data/zarr_test/tick.dat'

    if not os.path.exists(memmap_path):
        console.print(f"[red]Could not find test data at {memmap_path}[/red]")
        return

    num_cols = len(col_to_ind_dict)

    # Calculate rows from file size
    file_size = os.path.getsize(memmap_path)
    num_ticks = file_size // (num_cols * 8) # 8 bytes per int64

    # Open the memmap in read-only mode
    data = np.memmap(memmap_path, dtype='i8', mode='r', shape=(num_ticks, num_cols))

    console.print(f"[green]Array contains {num_ticks:,} ticks, ready for zero-RAM overhead search.[/green]")

    # Generate 10,000 random entry points and targets
    np.random.seed(42)
    NUM_QUERIES = 10_000

    # Pick random starting indices (leaving enough room at the end of the array)
    start_indices = np.random.randint(0, num_ticks - 1_000_000, size=NUM_QUERIES)

    queries = []
    for idx in start_indices:
        start_price = data[idx, PRICE_COL]
        # Target TP and SL (e.g., +/- 10 to 50 points, remembering price is x100)
        tp_offset = np.random.randint(1000, 5000)
        sl_offset = np.random.randint(1000, 5000)
        queries.append((idx, start_price + tp_offset, start_price - sl_offset))

    console.print(f"\n[yellow]Running {NUM_QUERIES:,} Brute Force tick-by-tick searches...[/yellow]")
    t0 = time.perf_counter()
    bf_results = []
    for idx, tp, sl in queries:
        res = brute_force_search(data, idx, tp, sl)
        bf_results.append(res)

    bf_time = time.perf_counter() - t0
    console.print(f"Brute Force took: [red]{bf_time:.4f} seconds[/red] ({(bf_time/NUM_QUERIES)*1000:.3f} ms per query)")

    console.print(f"\n[yellow]Running {NUM_QUERIES:,} Numba Zarr hierarchical searches...[/yellow]")

    # Warm up Numba JIT (run once so compilation time isn't counted in the benchmark)
    search(data, queries[0][0], queries[0][1], queries[0][2])

    t0 = time.perf_counter()
    numba_results = []
    for idx, tp, sl in queries:
        res = search(data, idx, tp, sl)
        numba_results.append(res)
    numba_time = time.perf_counter() - t0
    console.print(f"Numba Zarr took: [green]{numba_time:.4f} seconds[/green] ({(numba_time/NUM_QUERIES)*1000:.3f} ms per query)")

    # Verify correctness
    mismatches = 0
    for i in range(NUM_QUERIES):
        if bf_results[i] != numba_results[i]:
            mismatches += 1

    if mismatches == 0:
        console.print("\n[bold green]SUCCESS: Both algorithms returned identical results 100% of the time![/bold green]")
    else:
        console.print(f"\n[bold red]ERROR: {mismatches} results did not match![/bold red]")

    speedup = bf_time / numba_time
    console.print(f"\n[bold cyan]Numba Zarr is {speedup:.1f}x FASTER than Brute Force![/bold cyan]")

if __name__ == "__main__":
    main()