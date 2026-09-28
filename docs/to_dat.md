# Step 2 — `data_pipeline/to_dat.py`

Turns step 1's tick file into **`tick.dat`**, every tick with its bar summaries, which the stop search
(`stop_search.StopSearch`) walks, and the **OHLCV bar files** `{freq}_ohlcv.parquet`, which feature engineering
(step 3) and the backtest read. The 1-min bar file also carries the **limit-entry columns**, used only to execute
trades (see *Limit-entry columns*).

```bash
python -m data_pipeline.to_dat                                                   # full data, default input and output
python -m data_pipeline.to_dat --src data/ES_trades_2024.parquet --out data/processed_2024
python -m data_pipeline.to_dat --limit 5 --out data/processed_test               # first 5 sessions, e.g. for the benchmarks
```

Run from the repo root. Daily sessions are added by `python -m data_pipeline.daily_update append`, which reuses
`process_session` and appends to `tick.dat` in place (see `data_pipeline/README.md`).

| Option | Meaning |
|---|---|
| `--src` | Step 1's output, default `data/ES_trades_concat.parquet` (must equal step 1's default; a test checks) |
| `--out` | Output folder, default `params.DATA_PATH` (`data/processed`), created if missing |
| `--limit N` | Only the first N sessions |
| `--start YYYY-MM-DD` | Skip sessions before that date (row groups that end before it are not read) |

**A run first deletes every `*.parquet`, `*.dat` and `*.tmp` in `--out`**, except `--src` and
`training_data.parquet`. Keep anything else out of that folder.

## Output

| File | Rows | What |
|---|---|---|
| `tick.dat` | one per tick (720,730,496 for 2020-01-02 → 2026-09-18) | Packed 92-byte rows, no header: `stop_search.params.DAT_DTYPE` (below). 66,307,205,632 bytes (66 GB) |
| `{freq}_ohlcv.parquet` for `1 5 10 15 30 60 day session` | one per bar with at least one trade (1-min: 2,373,007) | zstd parquet, columns below |

`session` bars group each date's ticks by step 1's `session` block: Asia 00:00-08:00, Europe 08:00-16:00 and US
16:00-23:00 on the shifted clock (18:00-02:00, 02:00-10:00 and 10:00-17:00 New York), three bars a date. They are in
the bar files only: `tick.dat` has no `session` columns, so its layout is unchanged.

### `tick.dat` columns

Each row is 92 bytes, little-endian: first the 12 uint32 columns (`INDEX_COLS`: `start_ind`, `ts`, then
`next_ind_{f}`), then the 21 uint16 ones (`PRICE_COLS`: `price`, then `high_{f}` and `low_{f}`), then 2 bytes of
padding (`_pad`, 0) so each row starts on a 4-byte boundary. Read it with `np.memmap(path, dtype=DAT_DTYPE)`;
`StopSearch` reads the two blocks as a uint32 table and a uint16 table of the same rows, and returns prices and row
numbers as int64 (unsigned values wrap when subtracted). The widths set the limits: prices up to 65,535 ticks
(16,383.75 ES points; fine for ES, not for NQ), row numbers up to 4.29 billion (about 115 million a year now), `ts` up
to 2106. `offset_session` refuses a value that does not fit rather than wrapping it. Until 2026-09-28 every column was
int64 (264-byte rows, 190 GB); `StopSearch.load` refuses such a file with a message to rebuild.

| Column | On which rows | Value |
|---|---|---|
| `start_ind` | the first tick of each 1-min bar | its own row number in the file (0 on other rows) |
| `ts` | every row | seconds since 1970 of the shifted clock (New York + 6h, read as if it were UTC) |
| `price` | every row | trade price in ticks (points x 4) |
| `high_{f}`, `low_{f}`, `next_ind_{f}` for `f` in `1 5 10 15 30 60 day 1s 15s 5s` | the first tick of each `f` bar | the bar's high and low, and the row just after its last tick (all 0 on other rows) |

Row numbers are global (row 0 = the first tick of the first session). `next_ind_{f} != 0` marks the first
tick of an `f` bar; the bar files' `start_ind` lists them. The order of `FREQS` in `stop_search/params.py`
sets the column order within each block. Changing `FREQS` changes the file layout, so rerun this step before using
`stop_search` again.

### Bar files

| Column | Type | Value |
|---|---|---|
| `start_ind` | int64 | `tick.dat` row of the bar's first tick |
| `ts` | timestamp | the bar's start on the shifted clock, whole seconds (parquet stores it in ms, always `.000`) |
| `open`, `close` | int32 | first and last tick's price (ticks) |
| `high`, `low` | int32 | highest and lowest tick |
| `volume` | int64 | contracts traded |
| `hl` | bool | the bar's high came before its low (first occurrence of each) |
| `rth` | bool | any tick in regular hours |
| `session`, `hour` | int8, int64 | the first tick's session block and hour |
| `news_fomc` … `news_gdp` | int8 | 1 if any tick is in that event's window |
| 1-min only: `pre_high_1`, `pre_low_1`, `fill_px_{long,short}_1`, `after_high_{long,short}_1`, `after_low_{long,short}_1` | int64 | the limit-entry columns, below |

## How bars are built

- **Sessions**: step 1's rows grouped by the date of `ts` (the shifted clock puts each Globex session on one
  date). Check: every tick is inside `[date 00:00, date 23:00)`.
- **Bar edges** are counted from the session's 00:00 (New York 18:00) on step 1's whole-second `ts`: a 60-min
  bar is 00:00–00:59:59, 01:00–01:59:59, …; `day` is the whole session. Every bar size divides the ones above it
  (`stop_search.params.CHILD`), so a bar's first tick is also the first tick of its first child bar.
- **Only bars with trades exist.** A minute with no trade has no 1-min bar; nothing is filled forward.
- Prices stay in ticks (uint16 in `tick.dat`, int32 in the bar files).
- Sessions run in parallel (up to 24 processes, at most twice that many in flight) and are written in session
  order, so row order is time order.
- `tick.dat` is written to `tick.dat.tmp` and renamed when the run finishes, so an existing `tick.dat` is always
  complete. The bar files are written in place, about 1 million bars at a time.

## How the stop search uses `tick.dat`

`StopSearch.first_hit(freq, start, upper, lower)` asks which level the `freq` bar starting at row `start` touches
first: 1 upper, -1 lower, 0 neither. It reads the bar's `high_{freq}` / `low_{freq}` on its first row: if only one
level is inside the bar, that decides it; if neither, the answer is 0 (the backtest then asks about the next bar);
if both, it splits the bar into its child bars (`stop_search.params.CHILD`, e.g. `day → 60 → 30 → 10 → 5 → 1 →
15s → 5s → 1s`), checks them in time order through `next_ind`, and walks a 1-second bar holding both tick by tick.
So most answers come from a few rows, and the tick walk is rare. Details: `stop_search/stop_search.py`.

## Limit-entry columns

A limit order at a 1-min bar's **open** `p`, sent when the minute starts, reaches the market
`stop_search.params.ENTRY_LATENCY_MS` (300 ms) later. Per minute, on the minute's own row:

| Column | Value |
|---|---|
| `pre_high_1`, `pre_low_1` | highest / lowest trade **during the latency** (before arrival), the open included |
| `fill_px_long_1` | a **buy** limit at `p`: the first trade after arrival if it is at or below `p` (filled at once, at that price); if it is above, `p` when a later trade comes back to `p` or below within the minute; else **0** (no fill) |
| `after_high_long_1`, `after_low_long_1` | highest / lowest trade from arrival **until the fill**, the filling trade excluded (the fill price itself when filled at once; to the minute's end when there is no fill) |
| `fill_px_short_1`, `after_high_short_1`, `after_low_short_1` | the same for a **sell** limit: "better" is at or above `p` |

300 ms is the trading machine's order latency. The columns are computed here, so a different latency means
editing `ENTRY_LATENCY_MS` and rerunning this step (the backtest reads the new columns as they are).

A minute with no trade after arrival gets fill 0 and after-high/low `p`. The arrival time uses step 1's
nanosecond `ts_ns`; a step-1 file without it (e.g. sessions appended from IB, whole seconds only) counts every
trade in the minute's first second as during the latency.

**Not shifted, not features.** Each minute's columns describe that minute itself. They are used only to execute a
trade that enters in that minute: `backtesting/entry.py` looks them up by the entry bar's `start_ind` and checks
whether the TP / SL level was reached before the fill (see `docs/backtesting.md`). Feature engineering drops them,
so they never reach the training data or a signal.

Over all 2,373,007 minutes, a buy limit at the open:

| On arrival | Minutes | Result |
|---|---|---|
| Price at the open | 1,811,497 | filled at the open |
| Price below the open (better) | 283,213 | filled at that price |
| Price above the open, came back | 244,339 | filled at the open |
| Price above the open, never came back | 33,906 | not filled |
| No trade after 300 ms | 52 | not filled |

98.6% filled: 87.9% of the fills at the open and 12.1% better. A sell is the mirror image, and no fill is ever
worse than the open.

**The fill rule is optimistic.** It counts a trade at `p` as a fill for both the buy and the sell, but only one side
is marketable at a time; the other waits at the back of the queue and should fill only when price trades through
`p`. On 20 recent raw days, a stricter rule using the raw aggressor `side` fills about 89% of minutes against this
rule's 98%. Step 1 drops `side`; keeping it would allow the stricter rule.

## Checks that stop the run

| Check | Catches |
|---|---|
| Every tick of a session in `[date 00:00, date 23:00)` | Clock mistakes, a tick on the wrong date |
| `StopSearch.load`: file size a whole number of 33-column rows | A cut or wrong-layout `tick.dat` |
| `StopSearch.first_hit`: a bar summary contradicting its ticks raises | A broken build |
| Backtest: `tick.dat` price at each entry equals the training data's open | Files from different builds |

## Verified

- Rebuild with the limit-entry columns (2026-09): all 33 `tick.dat` columns hash-identical to the build before;
  the 5-min to daily bar files identical; `1_ohlcv.parquet`'s old columns identical, plus the 8 new ones; the
  training data still aligned.
- Every 1-min open equals `tick.dat`'s price at its `start_ind`; every 5, 15 and 60-min bar starts on a minute and
  its open equals that minute's open.
- `tests/test_entry.py`: hand-worked minutes, 5 random sessions of 20,000 ticks against a tick-by-tick reference, a
  whole-second file, and a spike during the latency (the limit entry is skipped, a market entry is not).
- 3,000 random real 5-min entries matched a direct calculation from `tick.dat` prices and step 1's `ts_ns`.
- `tests/test_stop_search.py`, `tests/test_daily_update.py` (a full build equals a build plus `append`),
  `scripts/sample_output.py` (every bar against its ticks) and `benchmarks/` (`first_hit` against a tick scan).

## Known limitations

- No empty bars: indicators over N bars span more than N minutes across quiet stretches.
- `hl` uses the first occurrence of the high and the low; a bar whose high and low are the same tick counts as
  low-first.
- The limit-entry fill rule is optimistic (above).
- The end-of-run message extrapolates to 1,684 sessions (an old count); the data now has 1,737.
