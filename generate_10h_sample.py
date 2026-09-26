import polars as pl
import pandas as pd
import os

# Create sample directory if it doesn't exist
os.makedirs('data/processed/sample', exist_ok=True)

# Load the full generated training data
df = pl.read_parquet('data/processed/training_data.parquet')

# Find the first timestamp in the dataset
start_ts = df.select(pl.col('ts').min()).item()

# Calculate the timestamp for 10 hours later (10 hours * 60 mins * 60 seconds)
end_ts = start_ts + (10 * 3600)

# Filter the dataframe for the first 10 hours
sample_df = df.filter(pl.col('ts') <= end_ts)

# Save the sample
output_path = 'data/processed/sample/feature_engineering_sample_10h.parquet'
sample_df.write_parquet(output_path)

print(f"Generated 10-hour sample with {sample_df.height} rows.")
print(f"Saved to: {output_path}")

# Display a quick snapshot with readable dates
df_pd = sample_df.select(['ts', 'open_1', 'close_1', 'open_15', 'close_15', 'open_60', 'close_60']).tail(10).to_pandas()
df_pd['dt'] = pd.to_datetime(df_pd['ts'], unit='s')
print("\nEnd of the 10-hour snapshot:")
print(df_pd[['dt', 'open_1', 'close_1', 'open_15', 'close_15', 'open_60', 'close_60']])
