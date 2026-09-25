import polars as pl
import pandas as pd

# Let's inspect the exact minute alignment between the 1m and 15m data in Polars
df = pl.read_parquet('data/processed/training_data.parquet')
df_pd = df.select(['ts', 'open_1', 'close_1', 'open_15', 'close_15']).head(25).to_pandas()
df_pd['dt'] = pd.to_datetime(df_pd['ts'], unit='s')

print(df_pd[['dt', 'open_1', 'close_1', 'open_15', 'close_15']].head(25))
