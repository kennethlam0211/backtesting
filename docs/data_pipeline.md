# Step 1 — `data_pipeline/raw_data_preprocessing.py`

Turns the raw Databento ES trade files (one per session) into **one clean, time-ordered tick file**
with one contract per session, prices in ticks, and a clock that looks the same in summer and winter.
Step 2 (`data_pipeline/to_dat.py`) reads its output and writes `tick.dat` (for the stop search, `stop_search.first_hit`)
and the OHLCV bar files (`open`/`high`/`low`/`close` in ticks, int32, like `price` here).

```bash
python -m data_pipeline.raw_data_preprocessing                      # all sessions -> data/ES_trades_concat.parquet
python -m data_pipeline.raw_data_preprocessing --start 2024-01-01 --end 2024-12-31 --out data/ES_trades_2024.parquet
```

Run from the repo root: `raw_data/` and the output path are relative to it, and the output folder
(`data/`) must already exist. `params/` is read from the repo root.

Full run: 1,737 sessions (2020-01-02 → 2026-09-18) in about 3 minutes; about 1 GB of RAM.

## Output

`data/ES_trades_concat.parquet` (zstd), one row per merged tick, in time order:

| Column | Type | Meaning |
|---|---|---|
| `ts` | timestamp, whole seconds | New York time + 6h (see *Clock*). Session = `ts.date` |
| `price` | int32 | Trade price **in ticks**: points x 4 (4000.25 -> 16001). Raw, not adjusted across rolls |
| `volume` | int64 | Contracts traded (sum of merged raw `size`) |
| `rth` | int8 | `1` in regular hours, New York `[09:30, 16:00)` = `ts` `[15:30, 22:00)` |
| `session` | int8 | 8-hour block of `ts`: `1` Asia = `ts` 00–07 (NY 18:00–02:00), `2` Europe = 08–15 (NY 02:00–10:00), `3` US = 16–22 (NY 10:00–17:00). The first 30 min of RTH are `2` |
| `hour` | int8 | Hour of `ts` (shifted clock, 0–22): `0` = NY 18:00, `15` = NY 09:00 |
| `news_fomc`, `news_nfp`, `news_cpi`, `news_ppi`, `news_gdp` | int8 | `1` if the tick is in `[event - 5m, event + 5m)`, else `0` (see *News flags*) |

The contract code (`pdt_code`) is looked up and checked for every session but not written.
Every other raw column (`ts_recv`, `instrument_id`, `side`, `sequence`, …) is dropped.

## Inputs

| Path | Used for |
|---|---|
| `raw_data/ES_c_0_trades_<date>.parquet`, `raw_data/ES_v_0_trades_<date>.parquet` | The sessions. The prefix only records which request fetched the day; each date exists once |
| `raw_data/roll_open_blocks/ES_c_1_open_block_<date>.parquet` | New contract's ticks for the opening hours of the 27 roll days |
| `raw_data/pdt_codes.csv` | `instrument_id` + date range → contract code |
| `params/news_events.yaml` | News release dates and times (see *News flags*); a missing file stops the run |

Only top-level `raw_data/ES_*_trades_*.parquet` files are read, minus `*_partial.parquet`; so
`raw_data/_superseded/`, `raw_data/_batch/`, `_manifest*.csv` and `_qc_report*.txt` are never touched.
`--start` / `--end` compare against the date in the file name.

## What happens to each session (in date order)

1. **Pick and sort** — top-level `ES_*_trades_*.parquet`, sorted by the **date in the name**
   (sorting by full name would put all `c_0` before all `v_0`; they interleave).
   Check: each date appears once.
2. **Schema check** — every file must have the first file's columns and types.
3. **Roll-day splice** (still in UTC) — see *Roll days*. Keeps only the new contract.
4. **One contract** — the session must now hold exactly one `instrument_id`.
5. **Contract code** — look up `pdt_code` in `raw_data/pdt_codes.csv` by number **and** date.
6. **Clock** — `ts_event` (UTC) → New York wall clock → **+6h** → tz-naive `ts`.
   Check: every tick is inside its file's session, `[date 00:00, date 23:00)`.
7. **Price to ticks** — `price x 4` as int32 (ES moves in 0.25-point ticks, so this is exact).
8. **Merge** — rows with the same **nanosecond** `ts` and the same price become one row,
   `volume = sum(size)`, kept in first-traded order. Check: total volume unchanged.
9. **Labels** — `rth`, `session` and `hour` from the (still nanosecond) `ts`.
10. **Floor to seconds** — after the merge, `ts` is floored to the second.
11. **News flags** — see *News flags*.
12. **Write** to the output file.

## Clock

Raw timestamps are UTC (nanoseconds). Converting to New York time first handles daylight saving;
adding 6h puts the Globex reopen at 00:00, so every session is exactly one calendar date.

| Event | New York | `ts` | HKT summer | HKT winter |
|---|---|---|---|---|
| Globex reopen | 18:00 (prev. day) | **00:00** | 06:00 | 07:00 |
| US regular open | 09:30 | 15:30 | 21:30 | 22:30 |
| US regular close | 16:00 | 22:00 | 04:00 | 05:00 |
| Globex close | 17:00 | **23:00** | 05:00 | 06:00 |

`ts` is the same all year; HKT is `ts + 6h` in summer and `ts + 7h` in winter.
No trading happens on a Sunday 2 am daylight-saving switch, so the conversion is never ambiguous.

## Roll days

Databento's continuous series switches contract at **00:00 UTC = 08:00 HKT**, 1–2 hours after the open.
So a raw roll-day file opens on the **old** contract. The block file holds the **new** contract for
exactly that gap, bought separately as `ES.c.1`.

```
                 open (06:00 HKT summer / 07:00 winter)      00:00 UTC = 08:00 HKT
raw file         old contract ────────────────────────────────┤ new contract ───────────►
block            new contract ────────────────────────────────┤
after splice     new contract (block) ────────────────────────┤ new contract (raw) ─────►
ts               00:00                                          02:00 summer / 01:00 winter
```

Splice = block rows + the raw file's new-contract rows; the old contract's rows are dropped.
It runs on UTC timestamps, before the clock conversion, because both files share that clock.
Checked on all 27 roll days: the block starts exactly at the open, ends before the raw file's new-contract
rows, is the same contract, and price moves at most 1 tick across the seam.

## Why these choices

- **Merge, never drop duplicates.** Identical raw rows (~0.5–1%) are separate fills of one aggressive
  order. Summing keeps volume exact; dropping would lose ~0.3%.
- **Merge before flooring to seconds.** Only truly simultaneous fills at one price are combined.
  After flooring, several rows can share a second and price; they stay separate rows, and **row order is
  the true trade order** (the stop search, `stop_search.first_hit`, walks rows in order).
- **Prices in ticks (x4).** One unit is one 0.25-point tick, so prices, OHLC bars and stop/target levels
  downstream are all whole numbers of ticks (10 points = 40). There is no grid check: a price off the
  0.25 grid would be truncated, which ES data does not have.
- **`ts_event`, not `ts_recv`.** Exchange match time is "when it happened"; `ts_recv` is dropped.
- **Contract code by number + date.** Databento `instrument_id`s are just numbers and can be reused.

## Contracts (`raw_data/pdt_codes.csv`)

28 contracts, `ESH0` → `ESZ6`. Built from the roll sequence (quarterly H/M/U/Z, starting from `ESH0` on
2020-01-02) and checked: every contract's last session is 3 or 7 days before its expiry (third Friday).
A contract after `ESZ6` needs a new row, or Step 1 stops with *"no contract code"*.

## News flags

Added in Step 1 to allow fast filtering later. Five `int8` columns (`news_fomc`, `news_nfp`, `news_cpi`, `news_ppi`, `news_gdp`).

- **Announcement times**: read from `params/news_events.yaml` when the script is imported. Each event type
  has its `dates` and one `time_et`; a date listed under `times:` uses its own time instead
  (FOMC `2020-03-03: '10:00'`, `2020-03-15: '17:00'`). An override for a date not in `dates` is ignored.
- **Clock**: the YAML times are New York local time, so only the +6h shift is added (no timezone
  conversion; daylight saving is already in the New York time).
- **Window**: `[event - 5m, event + 5m)` on the floored-second `ts`: a tick at `event - 5m` is flagged,
  one at `event + 5m` is not.
- **Speed**: `np.searchsorted` per session flags only the windows overlapping that session.

### `params/news_events.yaml`

Corrected against federalreserve.gov, bls.gov and bea.gov on 2026-09-24 (54 changes: 24 moved dates or
times, including the two FOMC time overrides, 25 added and 5 removed). The list of changes with sources
was removed once applied; it is in git: `git show cea32ac:params/news_date_audit.csv`. Dates are sorted with no duplicates; only these five event types exist.

| Event | Time (NY) | Dates | Per year |
|---|---|---|---|
| FOMC | 14:00 | 2019-01-30 → 2027-12-08 | 8; 9 in 2020 (emergency 03-03 10:00 and Sunday 03-15 17:00) |
| NFP | 08:30 | 2019-01-04 → 2026-09-04 | 12; 11 in 2025 (government shutdown) |
| CPI | 08:30 | 2019-01-11 → 2026-09-11 | 12; 11 in 2025 (government shutdown) |
| PPI | 08:30 | 2019-01-15 → 2026-09-10 | 12; 11 in 2025 (government shutdown) |
| GDP | 08:30 | 2019-02-28 → 2026-12-23 | 12 (11 in 2019) |

NFP, CPI and PPI end in September 2026: add the next releases before the data runs past them.

## Checks that stop the run

| Check | Catches |
|---|---|
| `params/news_events.yaml` exists | Silently all-zero news flags |
| Each session date once | A day downloaded twice (`c_0` and `v_0`) |
| Same schema as the first file | A changed download format |
| Block: one contract, same schema, ends before the raw new-contract rows | A wrong or overlapping block |
| One contract per session after the splice | A roll day without a block |
| Contract code found | A contract missing from `raw_data/pdt_codes.csv` |
| Every tick in `[date 00:00, date 23:00)` | Clock / daylight-saving mistakes, files on the wrong date |
| Volume unchanged by the merge | A merge bug |

## Verified

- All 1,737 expected sessions present (weekdays minus 15 exchange closures), none extra; raw row counts match.
  (1,684 from 2020-03-17 checked by an agent; the 53 early-2020 sessions 2020-01-02 → 2020-03-16 checked separately: all ESH0, same schema.)
- 2024 run: 259 sessions, 4 roll days spliced, 99,066,452 raw rows → 96,067,992 ticks, one contract per date,
  volume equal to the raw volume with the splice applied, `ts` in whole seconds and in time order.
- Adding news flags adds minimal overhead (33s total). 2024 checks: `news_fomc` 159,684 flagged ticks, `news_nfp` 156,036, `news_cpi` 152,784, `news_ppi` 92,433, `news_gdp` 56,395. Boundaries spot-checked manually (e.g. 2024-09-18 FOMC exactly `19:55:00` to `20:04:59`).

## Known limitations (left as is)

- **Thin sessions before each roll.** Databento switches only 3 days (since mid-2022; 7 days before)
  before expiry, while traders move about a week before. The 1–3 sessions before a roll day are on the
  fading old contract with 20–35% of normal ticks (e.g. 2023-12-11/12, 2025-09-15/16).
- **Prices are not back-adjusted.** Between the last old-contract session and the roll day, price jumps by the
  calendar spread (2024: +61 to +75 points). Fine for day trades closed at the session end; adjust before
  using indicators that span several sessions.
- **Holiday sessions** (e.g. July 4, Thanksgiving) are open but very thin.
- Sub-second timing is gone from the output (`raw_data` keeps the nanoseconds).
