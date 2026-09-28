import pandas as pd
import numpy as np
import os
import sys

# tick.dat's row layout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stop_search.params import DAT_DTYPE

def convert_dat_to_excel(dat_path, excel_path, limit=1000):
    print(f"Reading {dat_path}...")

    # Read the .dat file: packed rows of stop_search.params.DAT_DTYPE (uint32 / uint16 columns)
    data = np.fromfile(dat_path, dtype=DAT_DTYPE)

    # Limit rows
    if limit is not None:
        data = data[:limit]

    print(f"Loaded {len(data)} rows from dat. Converting to Excel without formatting ts...")

    # Convert to pandas dataframe
    df = pd.DataFrame(data).drop(columns='_pad')

    # Note: Intentionally NOT converting 'ts' to datetime here
    # to preserve raw unix timestamps.

    # Save to Excel
    df.to_excel(excel_path, index=False)
    print(f"Saved {dat_path} to {excel_path}")

def convert_parquet_to_excel(parquet_path, excel_path, limit=1000):
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Limit rows
    if limit is not None:
        df = df.head(limit)

    print(f"Loaded {len(df)} rows from parquet. Converting to Excel...")
    df.to_excel(excel_path, index=False)
    print(f"Saved {parquet_path} to {excel_path}")

if __name__ == "__main__":
    os.makedirs('data/processed/sample', exist_ok=True)

    # Convert tick.dat sample (raw unix timestamps)
    convert_dat_to_excel(
        'data/processed/sample/tick.dat',
        'data/processed/sample/tick_dat_sample.xlsx',
        limit=5000
    )

    # Convert 1_ohlcv sample (pandas parses parquet timestamps automatically)
    convert_parquet_to_excel(
        'data/processed/sample/1_ohlcv.parquet',
        'data/processed/sample/1_ohlcv_sample.xlsx',
        limit=5000
    )

def convert_training_data_to_excel():
    print(f"Reading data/processed/training_data.parquet...")
    df = pd.read_parquet('data/processed/training_data.parquet')
    
    # Save a sample of the first 5000 rows
    sample_path = 'data/processed/sample/training_data_sample.xlsx'
    print(f"Loaded {len(df)} rows. Converting first 5000 rows to Excel...")
    df.head(5000).to_excel(sample_path, index=False)
    print(f"Saved to {sample_path}")

if __name__ == "__main__":
    convert_training_data_to_excel()
