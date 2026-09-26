import polars as pl
import pandas as pd

# Load the parquet sample
df = pl.read_parquet('data/processed/sample/feature_engineering_sample_10h.parquet')

# Convert to pandas
df_pd = df.to_pandas()

# ts is the shifted clock (New York + 6h); next to it, the New York wall clock
df_pd.insert(1, 'datetime_NY', df_pd['ts'] - pd.Timedelta(hours=6))

# Save to Excel
excel_path = 'data/processed/sample/feature_engineering_sample_10h.xlsx'
df_pd.to_excel(excel_path, index=False)
print(f"Sample converted and saved to: {excel_path}")
