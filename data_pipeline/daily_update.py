"""
Daily update of the processed data: step 1 (raw_data_preprocessing) then step 2 (to_dat).

    python -m data_pipeline.daily_update init                         # once: full history from raw_data/ (Databento); rebuilds everything
    python -m data_pipeline.daily_update append                       # every day (cron): new sessions from MongoDB (IB recorder)
    python -m data_pipeline.daily_update append --until 2026-09-25    # up to a given session
    python -m data_pipeline.daily_update append --file data/fake_ib/ES_trades.jsonl --src ... --out ...   # from a file, e.g. fake data

init runs both steps on the full data with their default paths, overwriting their outputs.
append brings both outputs up to date (default paths, or --src / --out). Every weekday session after the
last one in step 1's file, up to --until (default: the last closed session), is read from MongoDB (see
ib_ticks.py) or from --file, converted with step 1's to_ticks and added to the end of step 1's file. Step 2
then adds every session of that file newer than tick.dat's last one to tick.dat and the bar files. So a
missed or failed run is caught up by the next one, and running it twice is safe. A weekday with no ticks
(an exchange holiday) is skipped.
Exit code 0 when the outputs are up to date; 1 when the --until session has no ticks or the append fails.
"""
import argparse
import contextlib
import os
import sys

import pandas as pd

from data_pipeline import ib_ticks
from data_pipeline import raw_data_preprocessing as step1
from data_pipeline import to_dat as step2

INIT_FIRST = "build the data first with `python -m data_pipeline.daily_update init`"


def init():
    """Full build: both steps with their defaults (all sessions in raw_data/, default paths)."""
    step1.main([])
    step2.main([])


@contextlib.contextmanager
def _lock(folder):
    """Stop two appends running at once (e.g. overlapping cron runs). No-op where fcntl is missing."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(os.path.join(folder, ".append.lock"), "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another append is running") from None
        yield


def append(collection, until=None, step1_out=step1.DEFAULT_OUT, step2_out=step2.DEFAULT_OUT, log=print):
    """
    Bring both outputs up to session `until` (default: the last closed one) from `collection` (MongoDB, or an
    ib_ticks.LocalCollection). Returns the weekdays skipped because the collection had no ticks for them.
    """
    until = ib_ticks.last_closed_session() if until is None else pd.Timestamp(until).date()
    for path in (step1_out, os.path.join(step2_out, "tick.dat")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; {INIT_FIRST}")
    empty = []

    with _lock(step2_out):
        last = step1.last_session(step1_out)
        if last is None:
            raise ValueError(f"{step1_out} holds no ticks; {INIT_FIRST}")

        def fetched():
            for date in ib_ticks.session_dates(last, until):
                raw = ib_ticks.fetch_session(collection, date)
                if raw.num_rows == 0:
                    log(f"{date}: no ticks, skipped")
                    empty.append(date)
                    continue
                day = step1.to_ticks(raw, pd.Timestamp(date), "IB")
                log(f"{date}: {raw.num_rows:,} trades -> {day.num_rows:,} ticks")
                yield day

        added1 = step1.append_sessions(step1_out, fetched())
        log(f"{step1_out}: {added1} session(s) added, last {step1.last_session(step1_out)}")
        added2 = step2.append_sessions(step2_out, step1_out)
        log(f"{step2_out}: {len(added2)} session(s) added, last {step2.last_session(step2_out)}")
    return empty


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build (init) or extend (append) the processed data.")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("init", help="full history from raw_data/ (Databento); overwrites all outputs")
    add = modes.add_parser("append", help="new sessions from MongoDB (IB recorder) added to the outputs")
    add.add_argument("--until", help="last session to add, YYYY-MM-DD (default: the last closed session)")
    add.add_argument("--file", help="read the trades from this JSON-lines file instead of MongoDB (e.g. fake_ib_data.py's output)")
    add.add_argument("--src", default=step1.DEFAULT_OUT, help="step 1's tick file (default: step 1's default output)")
    add.add_argument("--out", default=step2.DEFAULT_OUT, help="step 2's folder with tick.dat (default: step 2's default output)")
    args = parser.parse_args(argv)

    if args.mode == "init":
        init()
        return 0

    until = pd.Timestamp(args.until).date() if args.until else ib_ticks.last_closed_session()
    try:
        collection = ib_ticks.LocalCollection.from_file(args.file) if args.file else ib_ticks.connect()
        empty = append(collection, until, args.src, args.out)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"append: {e}", file=sys.stderr)
        return 1
    if until in empty:
        where = args.file or "MongoDB"
        print(f"append: no ticks for session {until} in {where} (exchange holiday, or the recorder did not run)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
