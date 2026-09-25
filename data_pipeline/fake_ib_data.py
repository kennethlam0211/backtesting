"""
Fake IB_recorder data, as if already pulled from MongoDB: the trades of the sessions after the last one in
step 1's file, so `daily_update append` can be tried without MongoDB or the recorder.

    python -m data_pipeline.fake_ib_data                                   # the 5 weekdays after step 1's last session
    python -m data_pipeline.fake_ib_data --days 3 --src data/test/ES_trades_concat.parquet

Writes one trade per line as JSON, in the recorder's document format (field names and time unit from
ib_ticks.FIELDS / TIME_UNIT), to --out (default data/fake_ib/ES_trades.jsonl):

    {"time": 1727208000, "price": 5100.25, "size": 3, "symbol": "ES"}

Read it with `python -m data_pipeline.daily_update append --file data/fake_ib/ES_trades.jsonl ...`, or load
it into a real MongoDB: `mongoimport --db ib --collection ES_trades --file data/fake_ib/ES_trades.jsonl`.
Prices are a random walk in 0.25 steps from step 1's last price, with trades denser in regular hours.
It is not market data: append it to a copy of the outputs, not to the real ones (see README).
"""
import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

from data_pipeline import ib_ticks
from data_pipeline import raw_data_preprocessing as step1

DEFAULT_OUT = "data/fake_ib/ES_trades.jsonl"

# Chance that a second has trades, and the mean number of extra trades in such a second
RTH_ACTIVE, RTH_EXTRA = 0.9, 3.0          # New York 09:30-16:00
OVERNIGHT_ACTIVE, OVERNIGHT_EXTRA = 0.25, 0.5


def fake_session(session_date, price, rng):
    """One session of fake trades: UTC epoch seconds, price in points, size. Returns them and the last price."""
    start, end = ib_ticks.session_window(session_date)
    secs = pd.date_range(start, end, freq="1s", inclusive="left")
    ny = secs.tz_convert(ib_ticks.NY)
    minute = ny.hour * 60 + ny.minute
    rth = (minute >= 9 * 60 + 30) & (minute < 16 * 60)

    active = rng.random(len(secs)) < np.where(rth, RTH_ACTIVE, OVERNIGHT_ACTIVE)
    trades = 1 + rng.poisson(np.where(rth, RTH_EXTRA, OVERNIGHT_EXTRA)[active])
    time = np.repeat(secs[active].as_unit("s").asi8, trades)
    steps = rng.choice([-1, 0, 1], size=len(time), p=[0.2, 0.6, 0.2])
    prices = price + 0.25 * np.cumsum(steps)
    sizes = rng.geometric(0.4, size=len(time))
    return time, prices, sizes, float(prices[-1]) if len(prices) else price


def json_time(sec):
    """A trade time as the recorder stores it (ib_ticks.TIME_UNIT)."""
    if ib_ticks.TIME_UNIT == "s":
        return int(sec)
    if ib_ticks.TIME_UNIT == "ms":
        return int(sec) * 1000
    # BSON date, as mongoexport writes it
    return {"$date": datetime.datetime.fromtimestamp(int(sec), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fake IB_recorder trades for the sessions after step 1's last one.")
    parser.add_argument("--src", default=step1.DEFAULT_OUT, help="step 1's tick file: the fake sessions follow its last session and price")
    parser.add_argument("--days", type=int, default=5, help="number of weekday sessions (default 5)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"JSON-lines file to write (default {DEFAULT_OUT})")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    last = step1.last_session(args.src)
    if last is None:
        raise SystemExit(f"{args.src} holds no ticks")
    price = step1.last_value(args.src, "price") / 4  # ticks -> points
    later = ib_ticks.session_dates(last, last + datetime.timedelta(days=args.days * 2 + 7))
    dates = [d for d in later if d not in ib_ticks.HOLIDAYS][:args.days]
    rng = np.random.default_rng(args.seed)
    f = ib_ticks.FIELDS

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    with open(args.out, "w") as out:
        for date in dates:
            time, prices, sizes, price = fake_session(date, price, rng)
            for t, p, s in zip(time, prices, sizes):
                out.write(json.dumps({f["time"]: json_time(t), f["price"]: float(p), f["size"]: int(s), f["symbol"]: ib_ticks.SYMBOL or "ES"}) + "\n")
            total += len(time)
            print(f"{date}: {len(time):,} fake trades, last price {price:.2f}")

    print(f"{total:,} trades for {dates[0]} .. {dates[-1]} -> {args.out}")
    print(f"next: python -m data_pipeline.daily_update append --file {args.out} --until {dates[-1]} --src {args.src} --out <its step 2 folder>")
    if os.path.abspath(args.src) == os.path.abspath(step1.DEFAULT_OUT):
        print("      --src is the real step 1 output: append the fake sessions to a copy instead (see data_pipeline/README.md)")


if __name__ == "__main__":
    main()
