import pandas as pd
import numpy as np

# Recreate the exact logic from reference/preprocessing_pandas.py
df_1m = pd.DataFrame({
    'ts': pd.date_range('2023-01-01 10:00:00', periods=60, freq='1min'),
    'close': range(1, 61)
})

def resampler(df): 
    if not df.empty:
        return pd.Series({
            'close': df['close'].iloc[-1]
        })

# line 1073: resample
df_15m = df_1m.groupby(pd.Grouper(freq='15min', key='ts'), group_keys=False).apply(resampler).reset_index()

# line 1136: merge
df_merged = df_1m.merge(df_15m, on='ts', how='outer', suffixes=('_1', '_15'))

# line 1140:
tem_df_cols = ['close_15']
target_index = df_merged[df_merged[tem_df_cols].notna().all(axis=1)].index
print(f"target_index initially: {target_index.tolist()}")

# line 1141:
target_index = [ind-1 for ind in target_index][2:]  
print(f"target_index after list comp: {target_index}")

# line 1143:
df_merged[tem_df_cols] = df_merged[tem_df_cols].ffill()

# line 1145:
df_merged.loc[~df_merged.index.isin(target_index), tem_df_cols] = pd.NA

# Later, in line 1184, the user drops all rows that have `open_15` etc which are effectively NA now!
# Let's look at what the user does in feature_preprocessing (line 917 to_stack):
# Actually, the user extracts array windows at `index_15`. Let's check `get_index` (line 1308).

def get_index(df):
    target_window = 600
    # The user iterates over df. For a row i, they look back `target_window` rows.
    # Then they `.dropna().tail(WINDOW_SIZE).index.tolist()`
    # Since we set almost everything to NA in line 1145, dropna() will only leave the indices in `target_index`!
    
    # Example for row 50:
    chunk = df['close_15'].iloc[max(0, 50-target_window):50+1].dropna()
    print(f"\nRow 50 sees these non-NA HTF values:\n{chunk}")
    
get_index(df_merged)

