import polars as pl
import numpy as np

# Test how polars handles casting float with NaN to Int32
arr = np.array([31066.0, np.nan, 29296.0], dtype=np.float64)
s = pl.Series("test", arr)
print("Original series:", s)
s = s.cast(pl.Int32)
print("Casted series:", s)
