import pandas as pd
import numpy as np

# Let's verify what timestamp index 29 and 44 actually are in the 1-minute dataframe!
df_1m = pd.DataFrame({
    'ts': pd.date_range('2023-01-01 10:00:00', periods=60, freq='1min'),
    'close': range(1, 61)
})
print("Index 29 in 1-minute df:")
print(df_1m.iloc[29])

print("\nIndex 44 in 1-minute df:")
print(df_1m.iloc[44])
