import time
import os
import numpy as np
from rich.console import Console
from search import search, col_to_ind_dict, PRICE_COL

console = Console()

def brute_force_search(data: np.ndarray, start_idx: int, target_high: int, target_low: int):
    n = data.shape[0]
    for i in range(start_idx, n):
        price = data[i, PRICE_COL]
        if price >= target_high:
            return i, 1
        if price <= target_low:
            return i, -1
    return None, None

def main():
    console.print("[cyan]Opening memory-mapped .dat array...[/cyan]")
    memmap_path = 'data/zarr_test/tick.dat'

    if not os.path.exists(memmap_path):
        console.print(f"[red]Could not find test data at {memmap_path}[/red]")
        return

    num_cols = len(col_to_ind_dict)
    file_size = os.path.getsize(memmap_path)
    num_ticks = file_size // (num_cols * 8)

    data = np.memmap(memmap_path, dtype='i8', mode='r', shape=(num_ticks, num_cols))

    np.random.seed(42)
    NUM_QUERIES = 10_000
    start_indices = np.random.randint(0, num_ticks - 1_000_000, size=NUM_QUERIES)

    queries = []
    for idx in start_indices:
        start_price = data[idx, PRICE_COL]
        tp_offset = np.random.randint(1000, 5000)
        sl_offset = np.random.randint(1000, 5000)
        queries.append((idx, start_price + tp_offset, start_price - sl_offset))

    # Warm up Numba JIT
    search(data, queries[0][0], queries[0][1], queries[0][2])

    for i, (idx, tp, sl) in enumerate(queries):
        bf_res = brute_force_search(data, idx, tp, sl)
        numba_res = search(data, idx, tp, sl)

        if bf_res != numba_res:
            print("\n" + "="*50)
            print("MISMATCH FOUND!")
            print(f"Input Parameters:")
            print(f"  start_idx: {idx}")
            print(f"  target_high: {tp}")
            print(f"  target_low: {sl}")
            print(f"\nResults:")
            print(f"  Brute Force returned: {bf_res}")
            print(f"  Numba returned:       {numba_res}")

            end_idx = None
            if bf_res[0] is not None and numba_res[0] is not None:
                end_idx = max(bf_res[0], numba_res[0])
            elif bf_res[0] is not None:
                end_idx = bf_res[0]
            elif numba_res[0] is not None:
                end_idx = numba_res[0]
            else:
                end_idx = idx + 10  # fallback

            print("\nData Subset (start_idx to returned indices):")
            print("Index | Price")
            print("-" * 20)
            # Show data up to slightly past the mismatch
            for j in range(idx, end_idx + 2):
                if j < num_ticks:
                    price = data[j, PRICE_COL]
                    marker = " <-- START" if j == idx else ""
                    if bf_res[0] == j and numba_res[0] == j:
                        marker += " <-- BF & NUMBA MATCH"
                    elif bf_res is not None and bf_res[0] == j:
                        marker += " <-- BF RESULT"
                    elif numba_res is not None and numba_res[0] == j:
                        marker += " <-- NUMBA RESULT"

                    print(f"{j:<5} | {price}{marker}")

            print("="*50 + "\n")
            return

if __name__ == "__main__":
    main()