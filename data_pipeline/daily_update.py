"""
Daily update of the processed data: step 1 (raw_data_preprocessing) then step 2 (to_dat).

    python -m data_pipeline.daily_update init                         # once: full history from raw_data/ (Databento); rebuilds everything
    python -m data_pipeline.daily_update append                       # every day (cron): new sessions from MongoDB (IB recorder)
    python -m data_pipeline.daily_update append --until 2026-09-25    # up to a given (closed) session
    python -m data_pipeline.daily_update append --file data/fake_ib/ES_trades.jsonl --src ... --out ...   # from a file, e.g. fake data

init runs both steps on the full data with their default paths, overwriting their outputs.
append brings both outputs up to date (default paths, or --src / --out). Every session after the last one in
step 1's file, up to --until (default: the last closed session), is read from MongoDB (see ib_ticks.py) or
from --file, converted with step 1's to_ticks and added to the end of step 1's file. Step 2 then adds every
session of that file newer than tick.dat's last one to tick.dat and the bar files. So a missed or failed run
is caught up by the next one, and running it twice is safe.
A weekday with no ticks is skipped if it is in params/holidays.yaml; otherwise the append stops there (the
sessions before it are kept): the recorder missed it, or it is a holiday missing from the yaml.
Exit code 0 when the outputs are up to date to --until; 1 otherwise.
"""
import argparse
import contextlib
import datetime
import os
import sys

import pandas as pd

from data_pipeline import ib_ticks
from data_pipeline import raw_data_preprocessing as step1
from data_pipeline import to_dat as step2

INIT_FIRST = "build the data first with `python -m data_pipeline.daily_update init`"


@contextlib.contextmanager
def _lock(step1_out):
    """
    One init or append at a time on the same data (e.g. a manual init while the cron append runs): a lock
    file next to step 1's output. No-op where fcntl is missing.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    os.makedirs(os.path.dirname(step1_out) or ".", exist_ok=True)
    with open(step1_out + ".lock", "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another init or append is running") from None
        yield


def init():
    """Full build: both steps with their defaults (all sessions in raw_data/, default paths)."""
    with _lock(step1.DEFAULT_OUT):
        step1.main([])
        step2.main([])


def append(collection, until=None, step1_out=step1.DEFAULT_OUT, step2_out=step2.DEFAULT_OUT, log=print):
    """
    Bring both outputs up to session `until` (a date; default: the last closed one) from `collection`
    (MongoDB, or an ib_ticks.LocalCollection). Returns the first session with no ticks, where the append
    stopped, or None when the outputs are up to date to `until`.
    """
    until = until or ib_ticks.last_closed_session()
    for path in (step1_out, os.path.join(step2_out, "tick.dat")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; {INIT_FIRST}")
    missing = None

    with _lock(step1_out):
        last = step1.last_session(step1_out)
        if last is None:
            raise ValueError(f"{step1_out} holds no ticks; {INIT_FIRST}")

        def fetched():
            nonlocal missing
            for date in ib_ticks.session_dates(last, until):
                raw = ib_ticks.fetch_session(collection, date)
                if raw.num_rows == 0 and date in ib_ticks.HOLIDAYS:
                    log(f"{date}: holiday, no ticks, skipped")
                    continue
                if raw.num_rows == 0:
                    # Stop here: appending a later session would leave this one out for good
                    missing = date
                    return
                day = step1.to_ticks(raw, pd.Timestamp(date), "IB")
                log(f"{date}: {raw.num_rows:,} trades -> {day.num_rows:,} ticks")
                yield day

        added1 = step1.append_sessions(step1_out, fetched())
        log(f"{step1_out}: {added1} session(s) added, last {step1.last_session(step1_out)}")
        added2 = step2.append_sessions(step2_out, step1_out)
        log(f"{step2_out}: {len(added2)} session(s) added, last {step2.last_session(step2_out)}")
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build (init) or extend (append) the processed data.")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("init", help="full history from raw_data/ (Databento); overwrites all outputs")
    add = modes.add_parser("append", help="new sessions from MongoDB (IB recorder) added to the outputs")
    add.add_argument("--until", type=datetime.date.fromisoformat, help="last session to add, YYYY-MM-DD (default: the last closed session)")
    add.add_argument("--file", help="read the trades from this JSON-lines file instead of MongoDB (e.g. fake_ib_data.py's output)")
    add.add_argument("--src", default=step1.DEFAULT_OUT, help="step 1's tick file (default: step 1's default output)")
    add.add_argument("--out", default=step2.DEFAULT_OUT, help="step 2's folder with tick.dat (default: step 2's default output)")
    args = parser.parse_args(argv)

    if args.mode == "init":
        init()
        return 0

    closed = ib_ticks.last_closed_session()
    until = args.until or closed
    try:
        if args.file:
            collection = ib_ticks.LocalCollection.from_file(args.file)
        elif until > closed:
            # The recorder is still writing it: appended now, the rest of the session would never follow
            raise ValueError(f"session {until} has not closed yet (17:00 New York); the last closed one is {closed}")
        else:
            collection = ib_ticks.connect()
        missing = append(collection, until, args.src, args.out)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"append: {e}", file=sys.stderr)
        return 1
    if missing:
        print(f"append: no ticks for session {missing} in {args.file or 'MongoDB'}; stopped there. Either the recorder "
              f"missed it (fix its data, then rerun), or it was a holiday (add it to params/holidays.yaml, then rerun)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
