"""
Step 2 on a sample of step 1's output, to check it by eye: the first --rows ticks of ES_trades_concat.parquet
go through to_dat exactly as in a full build (check_session, process_session, offset_session, bars_table),
and the resulting tick.dat and {freq}_ohlcv.parquet are shown, saved and checked against those ticks.
Run from the repo root:

    python -m scripts.sample_output                                  # the first 10,000 ticks, 1-min bars
    python -m scripts.sample_output --rows 50000 --freq 15
    python -m scripts.sample_output --src data/ES_trades_2024.parquet --show 20

Writes to --out (default data/sample/; data/processed is not touched): tick.dat and the bar files (so
StopSearch.load('data/sample/tick.dat') and pd.read_parquet work on them), and as CSV (opens in Excel)
step1_ticks.csv (the input), tick_dat.csv (every row and column, ts as a timestamp) and {freq}_ohlcv.csv.
Prices are in ticks (price x4, 4000.25 -> 16001). Times are the shifted clock (New York + 6h); tick.dat
stores them as seconds. The last bar can be cut short by the sample's end. Exit code 1 if a check fails.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data_pipeline import to_dat as step2
from stop_search import DAT_COLS, PARQUET_FREQS

COL = {c: i for i, c in enumerate(DAT_COLS)}


def bar_length(freq):
    return pd.Timedelta(days=1) if freq == "day" else pd.Timedelta(minutes=int(freq))


def build(ticks):
    """Step 2 on the ticks, session by session as to_dat does: tick.dat rows and each freq's bar table."""
    rows, bars, offset = [], {f: [] for f in PARQUET_FREQS}, 0
    for date, sess in ticks.groupby(ticks["ts"].dt.date, sort=True):
        sess = sess.reset_index(drop=True)
        step2.check_session(sess, date)
        tick_res, resampled, n = step2.process_session(sess, date)
        rows.append(step2.offset_session(tick_res, resampled, offset))
        for f in PARQUET_FREQS:
            if len(resampled[f]):
                bars[f].append(resampled[f])
        offset += n
    return np.concatenate(rows), {f: step2.bars_table(b) for f, b in bars.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run step 2 on a sample of step 1's output; show, save and check it.")
    parser.add_argument("--src", default=step2.DEFAULT_SRC, help="step 1's output (default: to_dat's default input)")
    parser.add_argument("--rows", type=int, default=10_000, help="ticks to take from the start of --src (default 10,000)")
    parser.add_argument("--freq", default="1", choices=PARQUET_FREQS, help="bar file to show and check (default 1)")
    parser.add_argument("--show", type=int, default=10, help="bars and tick.dat rows printed (default 10; the checks use all)")
    parser.add_argument("--out", default=os.path.join("data", "sample"), help="output folder (default data/sample)")
    args = parser.parse_args(argv)
    f = args.freq

    ticks = next(pq.ParquetFile(args.src).iter_batches(batch_size=args.rows)).to_pandas()
    dat, tables = build(ticks)

    os.makedirs(args.out, exist_ok=True)
    dat.tofile(os.path.join(args.out, "tick.dat"))
    for name, table in tables.items():
        pq.write_table(table, os.path.join(args.out, f"{name}_ohlcv.parquet"))

    # tick.dat as a table: every column, ts (seconds) shown as a timestamp
    tick_dat = pd.DataFrame(dat, columns=DAT_COLS)
    tick_dat["ts"] = dat[:, COL["ts"]].view("datetime64[s]")
    tick_dat.insert(0, "row", np.arange(len(dat)))
    bars = tables[f].to_pandas()
    bars["ts"] = bars["ts"].astype("datetime64[s]")

    same_rows = "tick.dat ts / price = the input ticks' ts / price, row for row"
    checks = {
        same_rows: [bool((tick_dat["ts"].to_numpy() == ticks["ts"].astype("datetime64[s]").to_numpy()).all()
                         and (dat[:, COL["price"]] == ticks["price"].to_numpy()).all())],
        "bar ts = its first tick's time, floored to the bar": [],
        "every tick is inside [ts, ts + 1 bar)": [],
        "open / close = first / last tick price": [],
        "high / low = highest / lowest tick price": [],
        "volume = the input ticks' volume summed over the bar": [],
        f"tick.dat high_{f} / low_{f} on the bar's first row = bar high / low": [],
        "the next bar starts where this one ends": [],
    }
    failed = {} if checks[same_rows][0] else {same_rows: ["some rows"]}
    for k, b in bars.iterrows():
        start = int(b["start_ind"])
        end = int(dat[start, COL[f"next_ind_{f}"]])  # the bar's end, from tick.dat's summary
        price = dat[start:end, COL["price"]]
        ts = tick_dat["ts"].iloc[start:end]
        results = [
            b["ts"] == (ts.iloc[0].normalize() if f == "day" else ts.iloc[0].floor(bar_length(f))),
            bool(((ts >= b["ts"]) & (ts < b["ts"] + bar_length(f))).all()),
            price[0] == b["open"] and price[-1] == b["close"],
            price.max() == b["high"] and price.min() == b["low"],
            ticks["volume"].iloc[start:end].sum() == b["volume"],
            (dat[start, COL[f"high_{f}"]], dat[start, COL[f"low_{f}"]]) == (b["high"], b["low"]),
            k + 1 == len(bars) or int(bars["start_ind"].iloc[k + 1]) == end,
        ]
        for name, ok in zip(list(checks)[1:], results):
            checks[name].append(bool(ok))
            if not ok:
                failed.setdefault(name, []).append(str(b["ts"]))

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(f"== {args.src}: the first {len(ticks):,} ticks ({ticks['ts'].iloc[0]} .. {ticks['ts'].iloc[-1]})\n")
        print(ticks.head(args.show).to_string(index=False))
        print(f"\n== {f}_ohlcv.parquet: {len(bars)} bars, the first {min(args.show, len(bars))}\n")
        print(bars.head(args.show).to_string(index=False))
        cols = ["row", "start_ind", "ts", "price", f"high_{f}", f"low_{f}", f"next_ind_{f}"]
        print(f"\n== tick.dat: {len(dat):,} rows x {len(DAT_COLS)} columns; the first {min(args.show, len(dat))} "
              f"(columns for {f}; tick_dat.csv has them all)\n")
        print(tick_dat[cols].head(args.show).to_string(index=False))

    print(f"\n== Checks on all {len(bars)} bars ({f}) and {len(dat):,} ticks")
    for name, oks in checks.items():
        print(f"  [{'ok' if all(oks) else 'FAIL'}] {name}" + (f"  ({', '.join(failed[name][:5])})" if name in failed else ""))

    ticks.to_csv(os.path.join(args.out, "step1_ticks.csv"), index=False)
    tick_dat.to_csv(os.path.join(args.out, "tick_dat.csv"), index=False)
    bars.to_csv(os.path.join(args.out, f"{f}_ohlcv.csv"), index=False)
    print(f"\nSaved in {args.out}/: tick.dat, " + ", ".join(f"{n}_ohlcv.parquet" for n in tables)
          + f", step1_ticks.csv, tick_dat.csv, {f}_ohlcv.csv")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
