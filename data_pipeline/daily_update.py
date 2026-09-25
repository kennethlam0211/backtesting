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
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_pipeline import ib_ticks
from data_pipeline import raw_data_preprocessing as step1
from data_pipeline import to_dat as step2
from stop_search.params import DAT_COLS, PARQUET_FREQS

INIT_FIRST = "build the data first with `python -m data_pipeline.daily_update init`"


# ---------------------------------------------------------------- step 1's file (to_ticks rows, one per tick)

def last_value(path, column):
    """The last tick's `column` in a step-1 file, or None if it holds no ticks. Reads only the last row group."""
    pf = pq.ParquetFile(path)
    for i in reversed(range(pf.num_row_groups)):
        values = pf.read_row_group(i, columns=[column]).column(column)
        if len(values):
            return values[-1].as_py()
    return None


def step1_last_session(path):
    """Session date of the last tick in a step-1 file, or None if it holds no ticks."""
    ts = last_value(path, 'ts')
    # ts is New York time + 6h, so its calendar date is the session
    return None if ts is None else ts.date()


def step1_append(path, days):
    """
    Add sessions to the end of the step-1 file. `days` holds to_ticks tables, one session each, oldest
    first, all newer than the file's last session; it can be a generator (append() reads MongoDB one
    session at a time). Parquet cannot be extended in place, so the file is copied to a temp file with the
    sessions added and swapped in; on any error the file is left as it was. Returns the number of sessions added.
    """
    pf = pq.ParquetFile(path)
    last = step1_last_session(path)
    days = (day for day in days if day.num_rows)
    first = next(days, None)
    if first is None:
        return 0
    tmp = str(path) + '.tmp'
    added = 0
    try:
        with pq.ParquetWriter(tmp, pf.schema_arrow, compression='zstd') as writer:
            for i in range(pf.num_row_groups):
                writer.write_table(pf.read_row_group(i))
            for day in itertools.chain([first], days):
                date = pc.min(day.column('ts')).as_py().date()
                if last is not None and date <= last:
                    raise ValueError(f"session {date} is not newer than the last one in {path} ({last}); run init to rebuild")
                writer.write_table(day.cast(pf.schema_arrow))
                last = date
                added += 1
        with open(tmp, 'rb') as f:
            os.fsync(f.fileno())  # on disk before it replaces the file
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    return added


# ---------------------------------------------------------------- step 2's folder (tick.dat and the bar files)

def _read_row(dat_path, i):
    """Row i of tick.dat (its DAT_COLS values), or an empty array past the end."""
    row_bytes = len(DAT_COLS) * 8
    with open(dat_path, 'rb') as f:
        f.seek(i * row_bytes)
        return np.frombuffer(f.read(row_bytes), dtype=np.int64)


def step2_last_session(out_dir):
    """Session date of the last row in out_dir/tick.dat, or None if the file is empty."""
    dat_path = os.path.join(out_dir, 'tick.dat')
    row_bytes = len(DAT_COLS) * 8
    size = os.path.getsize(dat_path)
    if size % row_bytes:
        raise ValueError(f"{dat_path}: {size} bytes is not a whole number of {len(DAT_COLS)}-column rows")
    if size == 0:
        return None
    last_ts = int(_read_row(dat_path, size // row_bytes - 1)[DAT_COLS.index('ts')])
    # ts is the shifted clock stored as if UTC, so its calendar date is the session
    return datetime.datetime.fromtimestamp(last_ts, datetime.timezone.utc).date()


def step2_committed_rows(out_dir):
    """
    Rows of out_dir/tick.dat the bar files account for: where the last day bar ends. A full build or a
    finished append leaves this equal to tick.dat's row count; an append killed part-way leaves it smaller.
    """
    day = pq.read_table(os.path.join(out_dir, 'day_ohlcv.parquet'), columns=['start_ind']).column('start_ind')
    if len(day) == 0:
        return 0
    row = _read_row(os.path.join(out_dir, 'tick.dat'), int(day[-1].as_py()))  # first row of the last session
    return int(row[DAT_COLS.index('next_ind_day')]) if len(row) else -1


def step2_append(out_dir=step2.DEFAULT_OUT, src=step2.DEFAULT_SRC):
    """
    Add every session of the step-1 file `src` newer than the last one in out_dir/tick.dat to tick.dat and
    the OHLCV bar files, exactly as a full build (to_dat.main) would have written them: same
    process_session / offset_session / bars_table. Returns the dates added (empty when tick.dat is already
    up to date). On any error tick.dat is cut back to its old size and the bar files are left as they were.
    """
    last = step2_last_session(out_dir)
    start = None if last is None else last + datetime.timedelta(days=1)
    dat_path = os.path.join(out_dir, 'tick.dat')
    size = os.path.getsize(dat_path)
    n_rows = size // (len(DAT_COLS) * 8)
    if step2_committed_rows(out_dir) != n_rows:
        raise ValueError(f"{dat_path} holds rows the bar files do not (an append was cut off?); "
                         f"rebuild step 2 from step 1's file with `python -m data_pipeline.to_dat`")

    added = []
    bars = {f: [] for f in PARQUET_FREQS}
    tmps = {}
    try:
        # Rows go straight onto the end of tick.dat; bars are collected and written once at the end
        with open(dat_path, 'r+b') as fh:
            fh.seek(size)
            for sess_date, sess_df in step2.iter_sessions(src, start):
                step2.check_session(sess_df, sess_date)
                tick_res, resampled_res, sess_len = step2.process_session(sess_df, sess_date)
                fh.write(step2.offset_session(tick_res, resampled_res, n_rows).tobytes())
                n_rows += sess_len
                for f in PARQUET_FREQS:
                    if len(resampled_res[f]) > 0:
                        bars[f].append(resampled_res[f])
                added.append(sess_date)
            if not added:
                return added

            # New bar files are written next to the old ones, and swapped in only once they and tick.dat are on disk
            for f in PARQUET_FREQS:
                path = os.path.join(out_dir, f'{f}_ohlcv.parquet')
                old = pq.read_table(path)
                tmps[path] = path + '.tmp'
                new = [step2.bars_table(bars[f]).cast(old.schema)] if bars[f] else []
                pq.write_table(pa.concat_tables([old, *new]), tmps[path], compression='zstd')
                with open(tmps[path], 'rb') as tmp:
                    os.fsync(tmp.fileno())
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with open(dat_path, 'r+b') as fh:
            fh.truncate(size)
        for tmp in tmps.values():
            if os.path.exists(tmp):
                os.remove(tmp)
        raise

    # day_ohlcv.parquet goes last: it is the commit record step2_committed_rows() checks
    for path in sorted(tmps, key=lambda path: path.endswith('day_ohlcv.parquet')):
        os.replace(tmps[path], path)
    return added


# ---------------------------------------------------------------- the two modes

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
        last = step1_last_session(step1_out)
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

        added1 = step1_append(step1_out, fetched())
        log(f"{step1_out}: {added1} session(s) added, last {step1_last_session(step1_out)}")
        added2 = step2_append(step2_out, step1_out)
        log(f"{step2_out}: {len(added2)} session(s) added, last {step2_last_session(step2_out)}")
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
