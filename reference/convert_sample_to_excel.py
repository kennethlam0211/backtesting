import polars as pl

# Load the parquet sample
df = pl.read_parquet('data/processed/sample/training_data.parquet')


# Save to Excel as is (ts is already a readable timestamp, the shifted clock)
excel_path = 'data/processed/sample/feature_engineering_sample_10h.xlsx'
df.to_pandas().to_excel(excel_path, index=False)

print(f"Sample converted and saved to: {excel_path}")
