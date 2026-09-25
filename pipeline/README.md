# Pipeline: how to run it

Three steps, run **in this order**, each from the **repo root** with `python -m` (so the packages
`pipeline` and `stop_search` import without any path setup):

```bash
pip install -r requirements.txt                 # once

python -m pipeline.raw_data_preprocessing       # 1. raw Databento trades -> one tick file
python -m pipeline.to_dat                       # 2. ticks -> tick.dat + OHLCV bar files
python -m pipeline.data_preprocessing           # 3. bar features (optional; preview only for now)
```

**Without arguments, each step runs on the full data and uses the default input and output paths set
in the script** (its `--src` / `--out` / `--data-dir` defaults): step 2 reads step 1's default output,
step 3 and `StopSearch.load()` read step 2's. Arguments are only needed for a date range, a subset, or
other paths.

```
raw_data/ES_*_trades_<date>.parquet ─┐
raw_data/roll_open_blocks/           ├─ 1 ─> tick parquet ─ 2 ─> tick.dat ───────────────> stop_search.StopSearch / first_hit
raw_data/pdt_codes.csv               │                           {freq}_ohlcv.parquet ─ 3 ─> features (printed)
params/news_events.yaml ─────────────┘
```

Each step reads the previous step's output, so after changing an earlier step, rerun every step after it.

**For now this is the one-time initial build of the whole history from the Databento files.** Every
run rebuilds its outputs from scratch (step 1 rewrites its parquet file, step 2 rewrites `tick.dat` and
every bar file); nothing is appended. Sessions from here on are recorded with the IB recorder
(`IB_recorder`); this pipeline does not read those recordings yet.

## 1. `raw_data_preprocessing.py` — raw trades to one tick file

```bash
python -m pipeline.raw_data_preprocessing                                   # full data, default output
python -m pipeline.raw_data_preprocessing --start 2024-01-01 --end 2024-12-31 --out data/ES_trades_2024.parquet
```

| | |
|---|---|
| Reads | `raw_data/ES_*_trades_<date>.parquet` (one per session), `raw_data/roll_open_blocks/ES_c_1_open_block_<date>.parquet`, `raw_data/pdt_codes.csv`, `params/news_events.yaml` |
| Writes | `--out` (default set in the script): one row per merged tick with `ts`, `price` (ticks, x4), `volume`, `rth`, `session`, `hour`, `news_*` |
| Options | `--start`, `--end`: first / last session date, inclusive (default: all sessions) |
| Before running | the output's folder must exist (`mkdir -p data`); the script does not create it |

Stops with an error on anything suspicious (changed schema, a roll day without a block, an unknown
contract, a tick outside its session, ...). Full details: [`docs/raw_data_preprocessing.md`](../docs/raw_data_preprocessing.md).

## 2. `to_dat.py` — ticks to `tick.dat` and bars

```bash
python -m pipeline.to_dat                                                   # full data, default input and output
python -m pipeline.to_dat --src data/ES_trades_2024.parquet --out data/processed_2024
python -m pipeline.to_dat --limit 5 --out data/processed_test               # first 5 sessions, e.g. for the benchmarks
```

| | |
|---|---|
| Reads | `--src` (default: step 1's default output) |
| Writes | into `--out` (default set in the script, created if missing): `tick.dat` (all ticks with their bar summaries, int64, columns = `stop_search.DAT_COLS`) and `{freq}_ohlcv.parquet` for `1 5 10 15 30 60 day` |
| Options | `--limit N`: only the first N sessions; `--start YYYY-MM-DD`: skip sessions before that date (default: all sessions) |
| Notes | Runs sessions in parallel (up to 24 processes). `tick.dat` is written as `tick.dat.tmp` and renamed only when the run finishes, so an existing `tick.dat` is always complete; the bar files are written in place |

Bar sizes and the `tick.dat` layout come from `stop_search/params.py`. Changing `FREQS` there changes
the file layout: rerun this step before using `stop_search` again. If you change the default `--out`,
update `DAT_PATH` in `stop_search/params.py` and `--data-dir` in `data_preprocessing.py` to match
(`pytest` fails until they agree).

## 3. `data_preprocessing.py` — bar features (preview)

```bash
python -m pipeline.data_preprocessing                                       # 1-min bars from step 2's default output
python -m pipeline.data_preprocessing --freq 5 --data-dir data/processed_2024
```

Reads `{--data-dir}/{--freq}_ohlcv.parquet` (defaults: step 2's default output folder, `1`), adds log
return, SMA 10 / 50, 20-bar volatility and high-low range with polars, and prints the table and the
shape of the RL state array. **It does not write a file yet.**

## Using the output

```python
from stop_search import StopSearch

stops = StopSearch.load()                                            # step 2's default tick.dat (params.DAT_PATH)
starts = stops.bar_starts('1')                                       # first tick of every 1-min bar
entry = stops.price(starts)
sides = stops.first_hit_many('1', starts, entry + 40, entry - 20)    # 1 upper first, -1 lower first, 0 neither
```

Pass a path to load another run, e.g. `StopSearch.load('data/processed_2024/tick.dat')`.
Prices are in ticks (x4): 10 points = 40.

## Checking the output

```bash
pytest                                                        # unit tests, no data needed
python -m pipeline.to_dat --limit 5 --out data/processed_test # the benchmarks read data/processed_test/tick.dat
python -m benchmarks.run_benchmark_mismatch 1                 # first_hit vs a tick-by-tick scan (also 5, 15, 60, day)
python -m benchmarks.benchmark_search 1                       # the same, with timings
```
