# Data pipeline: to do

What is left before `python -m data_pipeline.daily_update append` runs on real IB data every day.
How it all works: [README.md](README.md).

## Needs your input

- [ ] **MongoDB document format.** Send one real document from the IB recorder, then set `FIELDS` /
      `TIME_UNIT` in `ib_ticks.py` to match. The code expects one document per trade,
      `{"time", "price", "size", "symbol": "ES"}`, with `time` in UTC. Epoch seconds, epoch milliseconds
      and BSON dates already work. A time stored as text (e.g. `"2026-09-25T13:30:00Z"`) needs a small
      addition. A datetime saved in local time instead of UTC would put every tick in the wrong hour.
- [ ] **What the recorder saves.** Every trade (IB tick-by-tick `Last` / `AllLast`), not market-data
      snapshots. Otherwise volume and the tick path do not match the Databento days.
- [ ] **Contract.** The recorder must follow the front month and switch at a session open, as the
      Databento files do after the roll-day splice. Nothing checks this yet. If the documents carry the
      contract (e.g. `ESZ6`), add a one-contract-per-session check to `daily_update.append`.
- [ ] **`params/holidays.yaml`.** Replace the placeholder with the real list: a list of dates, or a
      mapping of date -> name. Until then, a holiday with no trades stops that day's run, and the
      message says to add it.
- [ ] **`params/news_events.yaml`.** Add the coming FOMC / NFP / CPI / PPI / GDP dates. The news
      flags are set when a day is appended, so a missing event means flags of 0 on that day.
- [ ] **Gap after the Databento history.** Databento ends on 2026-09-18, so `append` starts on
      2026-09-21 and stops at the first weekday with no ticks in MongoDB. Check that the recorder has
      every session from 2026-09-21 on. Fill any gap before the first run: more Databento files plus
      `init`, or the recorder's data.

## Setup on the machine

- [ ] `pip install -r requirements.txt` (`pymongo`).
- [ ] `MONGO_URI` / `MONGO_DB` / `MONGO_COLLECTION` (defaults `mongodb://localhost:27017`, `ib`, `ES_trades`).
- [ ] Index: `db.ES_trades.createIndex({symbol: 1, time: 1, _id: 1})`.
- [ ] First real run on a copy of the data, with `--src` / `--out` (as in the README's fake-data section).
      Then run it on the real outputs.
- [ ] Cron: `30 6 * * 2-6` on Hong Kong time, at least 30 minutes after the 17:00 New York close.
      The full line is in the README, under *Cron*.

## Not tested yet

- [ ] Against a real MongoDB. So far only a fake collection (`ib_ticks.LocalCollection`) and mongomock.
- [ ] On real IB data. If the recorder has days that Databento also covers (e.g. 2026-09-14 .. 18),
      compare one: tick count, volume, day OHLC, and `first_hit` on the same stops. Tick counts will be
      a little lower on IB days, since IB times are whole seconds.

## Later, if needed

- [ ] Every `append` rewrites step 1's file (`data/ES_trades_concat.parquet`), so the daily run takes
      longer as the history grows. If it gets slow, split step 1's output into one file per session or
      month.
- [ ] A past session cannot be changed in place: rerun `init`, then `append`.
