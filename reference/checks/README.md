# Checks: the pandas reference vs `data_pipeline/data_preprocessing.py`

One-off scripts from porting `reference/preprocessing_pandas.py` to polars. They print tables for you to
read; nothing is asserted. The same questions are now answered automatically by
`tests/test_data_preprocessing.py` (features vs the reference, and no look-ahead). Run them from the
repo root, e.g. `python reference/checks/prove_identical.py`.

On made-up 1-min bars (no data needed):

| Script | Shows |
|---|---|
| `verify_lookahead.py` | The reference's resample + merge + forward-fill: at 10:04 the 15-min column already holds the 10:14 close (look-ahead) |
| `test_user_pandas_bias.py` | The same, with the reference's `resampler` |
| `verify_user_pandas.py` | How the reference avoids it: the `target_index` mask, the `[2:]` slice, and what `get_index` then sees |
| `compare_approaches.py` | Which timestamps rows 29 and 44 are (the mask's rows) |
| `prove_identical.py` | The reference's masked values next to polars' shifted as-of join |

On the output of `python -m data_pipeline.data_preprocessing` (`data/processed/training_data.parquet`):

| Script | Shows |
|---|---|
| `compare_times.py` | The first 25 rows of the 1-min and 15-min columns |
| `verify_real_data.py` | The first 20 rows of the same |
| `verify_real_data_2.py` | The 1-min, 15-min, 60-min and day closes around an hour change and a day change (dates in 2020) |
