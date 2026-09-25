import argparse
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console

console = Console()

def generate_features(parquet_path: str | Path) -> pl.DataFrame:
    """
    Reads the OHLCV Parquet file and computes trading features/indicators using Polars.
    Returns a Polars DataFrame.
    """
    # Use scan_parquet for lazy evaluation (highly optimized, multi-threaded)
    lf = pl.scan_parquet(parquet_path)

    # 1. Basic Indicators & Feature Engineering
    # This executes purely in Rust and is fully vectorized.
    lf = lf.with_columns([
        # Log Returns (close / prev_close)
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"),

        # Simple Moving Averages
        pl.col("close").rolling_mean(window_size=10).alias("sma_10"),
        pl.col("close").rolling_mean(window_size=50).alias("sma_50"),

        # Volatility (Rolling standard deviation of log returns)
        (pl.col("close") / pl.col("close").shift(1)).log().rolling_std(window_size=20).alias("volatility_20"),

        # High-Low Range
        (pl.col("high") - pl.col("low")).alias("hl_range"),
    ])

    # 2. Drop rows with NaNs caused by the rolling windows
    lf = lf.drop_nulls()

    # Execute the lazy frame and load into RAM
    df = lf.collect()
    return df


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering for RL using Polars")
    parser.add_argument("--freq", type=str, default="1", help="Timeframe frequency to process (e.g. 1, 5, 15, day)")
    parser.add_argument("--data-dir", type=str, default="data/dat", help="Directory containing the OHLCV parquets")
    args = parser.parse_args()

    parquet_file = Path(args.data_dir) / f"{args.freq}_ohlcv.parquet"

    if not parquet_file.exists():
        console.print(f"[red]Error: Data file {parquet_file} does not exist. Run the pipeline first.[/red]")
        return

    console.print(f"[cyan]Loading and generating features for {parquet_file}...[/cyan]")

    # 1. Generate Features (Polars DataFrame)
    df = generate_features(parquet_file)
    console.print(f"[green]Generated {df.shape[0]} rows and {df.shape[1]} features.[/green]")
    console.print(df.head())

    # 2. Convert to NumPy for the RL Environment
    # We drop columns that RL doesn't need (like raw timestamps, unless you encode them as cyclic features)
    rl_columns = ["open", "high", "low", "close", "volume", "log_return", "sma_10", "sma_50", "volatility_20", "hl_range", "session", "hour", "rth"]

    # Check which columns exist before selecting to avoid KeyError
    available_cols = [c for c in rl_columns if c in df.columns]

    rl_state_array = df.select(available_cols).to_numpy()

    console.print(f"\n[yellow]RL State Array Shape: {rl_state_array.shape}[/yellow]")
    console.print(f"Sample RL Row (first row): {rl_state_array[0]}")

if __name__ == "__main__":
    main()
