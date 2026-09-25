# Step 1 — `preprocess_1_concat.py`

Turns the raw Databento ES trade files (one per session) into **one clean, time-ordered tick file**
with one contract per session and a clock that looks the same in summer and winter.
Step 2 (`to_zarr.py`) reads its output.

```bash
.venv/bin/python preprocess_1_concat.py                      # all sessions -> data/ES_trades_concat.parquet
.venv/bin/python preprocess_1_concat.py --start 2024-01-01 --end 2024-12-31 --out data/ES_trades_2024.parquet
```

Full run: 1,737 sessions (2020-01-02 → 2026-09-18) in about 3 minutes; about 1 GB of RAM.

## Output

`data/ES_trades_concat.parquet`, one row per merged tick, in time order:

| Column | Type | Meaning |
|---|---|---|
| `ts` | timestamp, whole seconds | New York time + 6h (see *Clock*). Session = `ts.date` |
| `price` | float | Trade price, raw (not adjusted across rolls) |
| `volume` | int64 | Contracts traded (sum of merged raw `size`) |
| `pdt_code` | string | Contract, e.g. `ESM4` |
| `news_fomc`, `news_nfp`, ... | int8 | `1` if tick is in `[event - 5m, event + 5m)`, else `0` (FOMC, NFP, CPI, PPI, GDP) |

Every other raw column (`ts_recv`, `side`, `sequence`, …) is dropped.

## Inputs

| Path | Used for |
|---|---|
| `raw_data/ES_c_0_trades_<date>.parquet`, `raw_data/ES_v_0_trades_<date>.parquet` | The sessions. The prefix only records which request fetched the day; each date exists once |
| `raw_data/roll_open_blocks/ES_c_1_open_block_<date>.parquet` | New contract's ticks for the opening hours of the 27 roll days |
| `config/pdt_codes.csv` | `instrument_id` + date range → contract code |

Ignored: `raw_data/_superseded/`, `raw_data/_batch/`, `_manifest*.csv`, `_qc_report*.txt`, `*_partial.parquet`.

## What happens to each session (in date order)

1. **Pick and sort** — top-level `ES_*_trades_*.parquet`, sorted by the **date in the name**
   (sorting by full name would put all `c_0` before all `v_0`; they interleave).
2. **Schema check** — every file must have the first file's columns and types.
3. **Roll-day splice** (still in UTC) — see *Roll days*. Keeps only the new contract.
4. **One contract** — the session must now hold exactly one `instrument_id`.
5. **Contract code** — look up `pdt_code` in `config/pdt_codes.csv` by number **and** date.
6. **Clock** — `ts_event` (UTC) → New York wall clock → **+6h** → tz-naive `ts`.
   Check: every tick is inside its file's session, `[date 00:00, date 23:00)`.
7. **Merge** — rows with the same **nanosecond** `ts` and the same price become one row,
   `volume = sum(size)`, kept in first-traded order. Check: total volume unchanged.
8. **Floor to seconds** — after the merge, `ts` is floored to the second.
9. **Write** to the output file.

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
  the true trade order** (the search in Step 3 walks rows in order).
- **`ts_event`, not `ts_recv`.** Exchange match time is "when it happened"; `ts_recv` is dropped.
- **Contract code by number + date.** Databento `instrument_id`s are just numbers and can be reused.

## Contracts (`config/pdt_codes.csv`)

28 contracts, `ESH0` → `ESZ6`. Built from the roll sequence (quarterly H/M/U/Z, starting from `ESH0` on
2020-01-02) and checked: every contract's last session is 3 or 7 days before its expiry (third Friday).
A contract after `ESZ6` needs a new row, or Step 1 stops with *"no contract code"*.

## News flags

Added in Step 1 to allow fast filtering later. Five `int8` columns (`news_fomc`, `news_nfp`, `news_cpi`, `news_ppi`, `news_gdp`).

- **Window**: Exactly 5 minutes before to 5 minutes after the announcement time (`[event - 5m, event + 5m)`). Ticks exactly at the boundary are excluded.
- **Announcement times**: Read from `config/news_events.yaml`.
- **Clock**: The YAML stores times in ET (e.g. 14:00). We convert this to our `ts` clock by adding 6 hours (no timezone math needed because our `ts` is already "New York wall clock + 6h").
- **Speed**: Vectorized parsing on import; `np.searchsorted` per session to only flag ticks inside windows overlapping that session. Very small overhead.

### Data quality findings from `config/news_events.yaml`

Reviewed on 2026-09-24:
- All dates are sorted and deduplicated per event type.
- **FOMC**: 21 dates out of range (before 2020-03-17 or after 2026-09-18). One weekend date found: `2020-03-15` (Sunday emergency cut). `2020-03-03` and `2025-08-22` are present. 8 scheduled per year, 9 in 2020 and 2025.
- **NFP/CPI/PPI**: 15 out of range each. 7 missing dates were added for each, and several corrected.
- **GDP**: 17 out of range. 2 missing dates added.
- **Other events**: The yaml contains no other event types. Only these five top-level keys exist.
- We updated the yaml directly with official dates after auditing against federalreserve.gov, bls.gov, and bea.gov. The 2020 entries (incl. 03-03, 03-15) will be useful when the backtest data expands to early 2020.

## Checks that stop the run

| Check | Catches |
|---|---|
| Same schema as the first file | A changed download format |
| Block: one contract, same schema, ends before the raw new-contract rows | A wrong or overlapping block |
| One contract per session after the splice | A roll day without a block |
| Contract code found | A contract missing from `config/pdt_codes.csv` |
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
