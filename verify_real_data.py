import polars as pl
import pandas as pd

# Load a chunk of the generated data to verify lookahead bias is fixed
df = pl.read_parquet('data/processed/training_data.parquet')

# Convert to pandas for easier display
df_pd = df.select(['ts', 'open_1', 'close_1', 'open_15', 'close_15']).head(50).to_pandas()
df_pd['dt'] = pd.to_datetime(df_pd['ts'], unit='s')

print(df_pd[['dt', 'open_1', 'close_1', 'open_15', 'close_15']].head(20))
