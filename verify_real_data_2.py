import polars as pl
import pandas as pd

df = pl.read_parquet('data/processed/training_data.parquet')
df_pd = df.select(['ts', 'open_1', 'close_1', 'open_15', 'close_15', 'open_60', 'close_60', 'open_day', 'close_day']).filter(
    (pl.col('ts') >= pl.datetime(2020, 1, 2, 0, 58).cast(pl.Int64) // 1000000) &
    (pl.col('ts') <= pl.datetime(2020, 1, 2, 1, 2).cast(pl.Int64) // 1000000)
).to_pandas()
df_pd['dt'] = pd.to_datetime(df_pd['ts'], unit='s')
print("--- 60m transition ---")
print(df_pd[['dt', 'close_1', 'close_15', 'close_60', 'close_day']])

df_pd_day = df.select(['ts', 'open_1', 'close_1', 'open_day', 'close_day']).filter(
    (pl.col('ts') >= pl.datetime(2020, 1, 3, 23, 58).cast(pl.Int64) // 1000000) &
    (pl.col('ts') <= pl.datetime(2020, 1, 4, 0, 2).cast(pl.Int64) // 1000000)
).to_pandas()
df_pd_day['dt'] = pd.to_datetime(df_pd_day['ts'], unit='s')
print("\n--- Day transition ---")
print(df_pd_day[['dt', 'close_1', 'close_day']])
