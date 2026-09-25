import pandas as pd
import polars as pl
import numpy as np

# 1. Create dummy 1m data
dates = pd.date_range('2023-01-01 10:00:00', periods=40, freq='1min')
df_1m = pd.DataFrame({'ts': dates, 'close_1': range(1, 41)})

# ==========================================
# 2. RUN USER PANDAS LOGIC
# ==========================================
def resampler(df): 
    if not df.empty:
        return pd.Series({'close_15': df['close_1'].iloc[-1]})

df_15m = df_1m.groupby(pd.Grouper(freq='15min', key='ts'), group_keys=False).apply(resampler).reset_index()
pandas_df = df_1m.merge(df_15m, on='ts', how='outer')

tem_df_cols = ['close_15']
target_index = pandas_df[pandas_df[tem_df_cols].notna().all(axis=1)].index
target_index = [ind-1 for ind in target_index][2:]  

pandas_df[tem_df_cols] = pandas_df[tem_df_cols].ffill()
pandas_df.loc[~pandas_df.index.isin(target_index), tem_df_cols] = pd.NA

# For comparison with Polars, let's ffill the single index 14 value forward just like Polars asof does
pandas_df['close_15_pandas'] = pandas_df['close_15'].ffill()

# ==========================================
# 3. RUN POLARS LOGIC
# ==========================================
# We have to convert timestamps to integer seconds because that's how tick.dat and Polars handles it now
pl_1m = pl.DataFrame(df_1m).with_columns(pl.col('ts').dt.timestamp('ms') // 1000)

# Simulate process_frequency(freq='15') but on datetime type to allow group_by_dynamic
pl_1m_dt = pl.DataFrame(df_1m)
pl_15m = pl_1m_dt.group_by_dynamic('ts', every='15m').agg(pl.col('close_1').last().alias('close_15_polars'))

# Convert back to int seconds
pl_15m = pl_15m.with_columns(pl.col('ts').dt.timestamp('ms') // 1000)

# The lookahead prevention shift (duration - 1m)
shift_seconds = (15 * 60) - 60
pl_15m = pl_15m.with_columns((pl.col('ts') + shift_seconds).alias('ts'))

# The backward join
polars_df = pl_1m.join_asof(pl_15m, on='ts', strategy='backward').to_pandas()
polars_df['ts'] = pd.to_datetime(polars_df['ts'], unit='s')

# ==========================================
# 4. COMPARE
# ==========================================
final_comparison = pandas_df[['ts', 'close_1', 'close_15_pandas']].copy()
final_comparison['close_15_polars'] = polars_df['close_15_polars']

print(final_comparison.iloc[10:35])
