import pandas as pd
import numpy as np

# Simulate 1-minute data
dates = pd.date_range('2023-01-01 10:00:00', periods=15, freq='1min')
df_1m = pd.DataFrame({
    'ts': dates,
    'close': range(1, 16)
})
print("--- 1-minute data (10:00 to 10:14) ---")
print(df_1m.head(5))
print("...")
print(df_1m.tail(2))

# Simulate what resampler did for 15-min
df_15m = df_1m.groupby(pd.Grouper(freq='15min', key='ts')).agg({'close': 'last'}).reset_index()
print("\n--- 15-minute grouped data ---")
print(df_15m)

# Simulate agg_df merge and ffill
df_merged = df_1m.merge(df_15m, on='ts', how='outer', suffixes=('_1m', '_15m'))
df_merged['close_15m'] = df_merged['close_15m'].ffill()

print("\n--- Merged result at 10:04 ---")
print(df_merged[df_merged['ts'] == '2023-01-01 10:04:00'])
print("Notice how close_15m is 15 (which is the price at 10:14). Lookahead bias!")
