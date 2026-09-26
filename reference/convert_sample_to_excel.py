import polars as pl
import pandas as pd

# Load the parquet sample
df = pl.read_parquet('data/processed/sample/training_data.parquet')

# Convert to pandas
df_pd = df.to_pandas()

# Convert the unix timestamp to human-readable datetime for easier viewing in Excel
df_pd.insert(1, 'datetime_NY', pd.to_datetime(df_pd['ts'], unit='s'))

# Save to Excel
excel_path = 'data/processed/sample/training_data.xlsx'
df_pd.to_excel(excel_path, index=False)
print(f"Sample converted and saved to: {excel_path}")
