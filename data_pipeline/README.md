# Data pipeline: how to run it

Two steps, run **in this order**, each from the **repo root** with `python -m` (so the packages
`data_pipeline` and `stop_search` import without any path setup). `daily_update.py` runs both, in one of two modes:

```bash
pip install -r requirements.txt                 # once

python -m data_pipeline.daily_update init       # once: the full history from the Databento files in raw_data/
python -m data_pipeline.daily_update append     # every day (cron): add the new sessions IB_recorder saved in MongoDB
```

`init` is the same as running the two steps by hand without arguments:

```bash
python -m data_pipeline.raw_data_preprocessing       # 1. raw Databento trades -> one tick file
python -m data_pipeline.to_dat                       # 2. ticks -> tick.dat + OHLCV bar files
```

**Without arguments, each step runs on the full data and uses the default input and output paths set
in the script** (its `--src` / `--out` defaults): step 2 reads step 1's default output, and
`StopSearch.load()` reads step 2's. `append` uses the same defaults. Arguments are only needed for a
date range, a subset, or other paths.

```
init    raw_data/ES_*_trades_<date>.parquet ─┐
        raw_data/roll_open_blocks/           ├─ 1 ─> tick parquet ─ 2 ─> tick.dat ──────────> stop_search.StopSearch / first_hit
        raw_data/pdt_codes.csv               │                           {freq}_ohlcv.parquet ─> features (template below)
        params/news_events.yaml ─────────────┘
append  MongoDB (IB_recorder) ─────────┐
        params/holidays.yaml           ├─ 1 (to_ticks) ─> + new sessions ─ 2 ─> + new rows and bars
        params/news_events.yaml ───────┘
```

Each step reads the previous step's output, so after changing an earlier step, rerun every step after it.

## Modes: `daily_update.py`

The append code lives in `daily_update.py`, not in the steps. It reuses step 1's `to_ticks` and step 2's
`process_session` (the code a full build runs), so the two steps stay standalone scripts.

### `init` — the whole history, once

Steps 1 and 2 on every session in `raw_data/`, with the default paths. Every output is rebuilt from
scratch: step 1 rewrites its parquet file, and step 2 rewrites `tick.dat` and every bar file. The sessions
that exist only in MongoDB are then gone from the outputs; run `append` right after `init` to add them back.
It catches up from the last Databento session, as long as MongoDB still holds those days.

### `append` — new sessions from MongoDB, every day

```bash
python -m data_pipeline.daily_update append                        # up to the last closed session (17:00 New York)
python -m data_pipeline.daily_update append --until 2026-09-25     # up to a given session (it must have closed)
python -m data_pipeline.daily_update append --file trades.jsonl    # from a JSON-lines file instead of MongoDB
```

1. Every weekday session after the last one in step 1's file, up to `--until` (default: the last
   closed session), is read from MongoDB one at a time (`ib_ticks.py`). Each is converted with step 1's
   `to_ticks` (same clock, ticks x4, merge, `rth` / `session` / `hour` / news flags) and added to the end
   of step 1's file. A weekday with no ticks is skipped if it is in `params/holidays.yaml`; any other
   weekday with no ticks stops the append there (see below).
2. Step 2 adds every session of step 1's file that is newer than `tick.dat`'s last one to `tick.dat` and
   the bar files. It writes exactly the rows a full build would: `tests/test_daily_update.py` checks that
   `init` on four sessions equals `init` on two plus `append`.

`--src` / `--out` point it at another step 1 file and step 2 folder (default: the steps' default outputs).

It is safe to run it again, and a failed run is caught up by the next one:

| Case | What happens |
|---|---|
| Already up to date | Nothing is read or written; exit 0 |
| A missed day (cron did not run) | Caught up by the next run |
| A holiday (in `params/holidays.yaml`) with no ticks | Skipped. A holiday that traded (a shortened session) is appended as usual |
| Any other weekday with no ticks | The append stops at that day and exits 1; the sessions before it are kept. Appending later days would leave it out for good. The message says what to do: the recorder missed it (fix its data, then rerun), or it is a holiday missing from the yaml (add it, then rerun) |
| `--until` a session that has not closed (17:00 New York) | Refused, exit 1: appended while still trading, the rest of the session would never follow. With `--file` any date is accepted |
| A bad `--until` (e.g. `2026-13-01`) | Usage error, exit 2 |
| Any error (MongoDB down, disk full, a price off the 0.25 grid, ...) | Files stay as they were. Step 1 writes a temp file and swaps it in; step 2 cuts `tick.dat` back and swaps in the new bar files only once they and `tick.dat` are on disk (fsync) |
| Step 2 fails after step 1 worked | The next run finishes step 2 from step 1's file, without MongoDB |
| A run killed part-way (power cut, OOM) | Can leave `tick.dat` longer than the bars. The next `append` detects it and stops; rebuild step 2 from step 1's file with `python -m data_pipeline.to_dat`, which keeps the appended sessions |
| Two runs at once (`init` or `append`, e.g. a manual `init` during the cron run) | The second stops: "another init or append is running" (lock file next to step 1's output) |
| A session older than the data | Never inserted. To change a past session, run `init` (then `append`) |

Cost: parquet cannot be extended in place, so each `append` rewrites step 1's file, streaming it one row
group at a time: memory stays flat, but the time grows with the length of the history.
Step 2 appends to `tick.dat` in place and rewrites the (small) bar files. If the daily run gets too
slow, step 1's output can become one file per session or month; not needed yet.

**Keep `params/holidays.yaml` up to date** (a placeholder for now): a list of dates, or a mapping of
date -> name. A holiday missing from it only stops the append on that day with a clear message; it
never loses data. Listing a shortened session does no harm either: it is appended if it has ticks.

**Keep `params/news_events.yaml` up to date** with the coming FOMC / NFP / CPI / PPI / GDP dates. The
news flags are set when a session is appended, so an event added to the yaml later only reaches that
session through `init` (then `append`).

#### MongoDB (`ib_ticks.py`)

Connection from environment variables: `MONGO_URI` (default `mongodb://localhost:27017`), `MONGO_DB`
(default `ib`) and `MONGO_COLLECTION` (default `ES_trades`). The script expects one document per trade:

```json
{"time": 1727208000, "price": 5100.25, "size": 3, "symbol": "ES"}
```

- Other field names: edit `FIELDS`.
- `time` stored as epoch milliseconds or as a BSON date: set `TIME_UNIT` to `"ms"` or `"datetime"`.
- `SYMBOL`: the documents to keep; set it to `None` to keep every document.
- Index: `db.ES_trades.createIndex({symbol: 1, time: 1, _id: 1})`. It serves the session query and its
  sort (time, then insertion order), so MongoDB neither scans the collection nor sorts a whole session in
  memory (which fails above 100 MB before MongoDB 6.0).
- Prices are rounded to the 0.25 grid (float noise such as `5100.2499999`); a price clearly off it (e.g.
  `5100.1`) stops the append with an error.

The recorder must save one contract per session and roll at a session open, as the Databento files
do after the splice. `append` does not check the contract. IB times are whole seconds, so same-price
trades within a second merge into one tick; Databento only merges trades in the same nanosecond. So IB
days have fewer rows per bar. The stop search is unaffected, because `ts` is whole seconds either way
and the first tick at each price keeps its place.

#### Trying it with fake data: `fake_ib_data.py`

`fake_ib_data.py` writes fake recorder data, as if already pulled from MongoDB, for the weekdays after
the last session in step 1's file. The prices are a random walk in 0.25 steps from the last real price,
with more trades in regular hours (about 100k trades a session). The file has one document per line, in
the recorder's format above: `append --file` reads it, and
`mongoimport --db ib --collection ES_trades --file data/fake_ib/ES_trades.jsonl` loads it into a real
MongoDB. **Append it to a copy of the outputs**: appended sessions cannot be taken out again without `init`.

```bash
mkdir -p data/test && cp data/ES_trades_concat.parquet data/test/ && cp -r data/processed data/test/processed

python -m data_pipeline.fake_ib_data --src data/test/ES_trades_concat.parquet               # 5 sessions -> data/fake_ib/ES_trades.jsonl
python -m data_pipeline.daily_update append --file data/fake_ib/ES_trades.jsonl --until <5th session, printed above> \
    --src data/test/ES_trades_concat.parquet --out data/test/processed
```

Then `StopSearch.load('data/test/processed/tick.dat')` reads the real sessions followed by the fake ones.
Options: `--days N` (default 5), `--seed`, `--out`.

#### Cron

Run it at least 30 minutes after the 17:00 New York close (so the recorder has written the last trades),
Monday to Friday New York time, from the repo root. The session
comes from New York time, so the machine's time zone only decides when cron fires. Cron does not load
your shell profile, so give the full path to the environment's Python. Example for a machine on Hong
Kong time (UTC+8): 06:30 HKT, Tuesday to Saturday, is 18:30 (summer) or 17:30 (winter) the evening
before in New York:

```cron
30 6 * * 2-6  cd /path/to/backtesting && MONGO_URI=mongodb://localhost:27017 .venv/bin/python -m data_pipeline.daily_update append >> data/append.log 2>&1
```

With cronie, `CRON_TZ=America/New_York` on the line above the entry lets you write the time in New York
time instead: `30 17 * * 1-5  cd /path/to/backtesting && ...`.

## 1. `raw_data_preprocessing.py` — raw trades to one tick file

```bash
python -m data_pipeline.raw_data_preprocessing                                   # full data, default output
python -m data_pipeline.raw_data_preprocessing --start 2024-01-01 --end 2024-12-31 --out data/ES_trades_2024.parquet
```

| | |
|---|---|
| Reads | `raw_data/ES_*_trades_<date>.parquet` (one per session), `raw_data/roll_open_blocks/ES_c_1_open_block_<date>.parquet`, `raw_data/pdt_codes.csv`, `params/news_events.yaml` |
| Writes | `--out` (default set in the script): one row per merged tick with `ts`, `price` (ticks, x4), `volume`, `rth`, `session`, `hour`, `news_*` |
| Options | `--start`, `--end`: first / last session date, inclusive (default: all sessions) |
| Notes | Creates the output's folder if missing |

Stops with an error on anything suspicious (changed schema, a roll day without a block, an unknown
contract, a tick outside its session, ...). Full details: [`docs/data_pipeline.md`](../docs/data_pipeline.md).

## 2. `to_dat.py` — ticks to `tick.dat` and bars

```bash
python -m data_pipeline.to_dat                                                   # full data, default input and output
python -m data_pipeline.to_dat --src data/ES_trades_2024.parquet --out data/processed_2024
python -m data_pipeline.to_dat --limit 5 --out data/processed_test               # first 5 sessions, e.g. for the benchmarks
```

| | |
|---|---|
| Reads | `--src` (default: step 1's default output) |
| Writes | into `--out` (default set in the script, created if missing): `tick.dat` (all ticks with their bar summaries, int64, columns = `stop_search.DAT_COLS`) and `{freq}_ohlcv.parquet` for `1 5 10 15 30 60 day` |
| Options | `--limit N`: only the first N sessions; `--start YYYY-MM-DD`: skip sessions before that date (default: all sessions) |
| Notes | Runs sessions in parallel (up to 24 processes). `tick.dat` is written as `tick.dat.tmp` and renamed only when the run finishes, so an existing `tick.dat` is always complete; the bar files are written in place |

Bar sizes and the `tick.dat` layout come from `stop_search/params.py`. Changing `FREQS` there changes
the file layout: rerun this step before using `stop_search` again. If you change the default `--out`,
update `DAT_PATH` in `stop_search/params.py` and `--data-dir` in `data_preprocessing(template).py` to
match (`pytest` fails until they agree).

## Template: `data_preprocessing(template).py` — bar features

Not a pipeline step: a starting point for bar features. The parentheses in the name keep it from being
imported, so run it by path:

```bash
python "data_pipeline/data_preprocessing(template).py"                      # 1-min bars from step 2's default output
python "data_pipeline/data_preprocessing(template).py" --freq 5 --data-dir data/processed_2024
```

Reads `{--data-dir}/{--freq}_ohlcv.parquet` (defaults: step 2's default output folder, `1`), adds log
return, SMA 10 / 50, 20-bar volatility and high-low range with polars, and prints the table and the
shape of the RL state array. It does not write a file.

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
pytest                                                        # unit tests (append against a fake MongoDB), no data needed
python -m data_pipeline.to_dat --limit 5 --out data/processed_test # the benchmarks read data/processed_test/tick.dat
python -m benchmarks.run_benchmark_mismatch 1                 # first_hit vs a tick-by-tick scan (also 5, 15, 60, day)
python -m benchmarks.benchmark_search 1                       # the same, with timings
```
