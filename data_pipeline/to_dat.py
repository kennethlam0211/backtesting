import argparse
import datetime
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.traceback import install

# Bar sizes and the tick.dat column layout are shared with the reader (stop_search.StopSearch)
from stop_search.params import FREQS, PARQUET_FREQS, DAT_COLS

install()
console = Console()

# Default input (must equal raw_data_preprocessing.DEFAULT_OUT; a test checks) and output folder
DEFAULT_SRC = "data/ES_trades_concat.parquet"
DEFAULT_OUT = "data/processed"


# One bar per row: the OHLCV parquet columns. ts is the bar's start on the shifted clock (a timestamp)
BAR_DTYPE = [
    ('start_ind', 'i8'), ('ts', 'M8[s]'), ('open', 'i4'),
    ('high', 'i4'), ('low', 'i4'), ('close', 'i4'),
    ('volume', 'i8'), ('hl', '?'), ('rth', '?'),
    ('session', 'i1'), ('hour', 'i8'),
    ('news_fomc', 'i1'), ('news_nfp', 'i1'), ('news_cpi', 'i1'), ('news_ppi', 'i1'), ('news_gdp', 'i1')
]


def to_unix_epoch(ts: pd.Series) -> pd.Series:
    """
    Converts naive datetimes to absolute Unix timestamps (in integer seconds)
    without unwinding or changing the timezone. We just treat the shifted clock
    as absolute time for storage.
    """
    return (ts.astype('datetime64[ns]').astype('int64') // 1_000_000_000)

def process_session(session_df: pd.DataFrame, session_date: datetime.date) -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
    """
    Process one session of tick data using vectorized numpy arrays.
    Returns:
        tick_res: DataFrame with 'ts', 'price', 'volume' and start_ind
        resampled_res: dictionary mapping freq string to structured numpy array of OHLCV bars
        sess_len: number of merged rows
    """
    # Ticks arrive already merged from step 1, ts in whole seconds; work in ns internally
    merged_ts = session_df['ts'].values.astype('datetime64[ns]').astype(np.int64)
    merged_vol = session_df['volume'].values.astype(np.int64)

    # Extract news flags from tick data
    merged_fomc = session_df['news_fomc'].values.astype(np.int8)
    merged_nfp = session_df['news_nfp'].values.astype(np.int8)
    merged_cpi = session_df['news_cpi'].values.astype(np.int8)
    merged_ppi = session_df['news_ppi'].values.astype(np.int8)
    merged_gdp = session_df['news_gdp'].values.astype(np.int8)

    # Extract RTH and session from tick data
    merged_rth = session_df['rth'].values.astype(bool)
    merged_session = session_df['session'].values.astype(np.int8)
    merged_hour = session_df['hour'].values.astype(np.int8)

    sess_len = len(merged_ts)

    # Price storage: step 1 stores price x4 (0.25-pt ticks) as int32; widened to int64 here for tick.dat.
    # Bar open/high/low/close go back to int32 in the OHLCV parquet files.
    merged_price = session_df['price'].values.astype(np.int64)

    tick_res = pd.DataFrame({
        'ts': session_df['ts'].values.astype('datetime64[s]'),  # New York + 6h; seconds only in tick.dat (offset_session)
        'price': merged_price,
        'start_ind': np.full(sess_len, -1, dtype=np.int64)
    })

    resampled_res = {}

    # Define midnight in ns for bar alignment
    # A session starts 00:00 on its date (step 1 put the Globex reopen there); bar edges count from it
    day0 = pd.Timestamp(session_date).value

    # tick.dat bar columns (high / low / next_ind per bar size): zero except on each bar's first row
    bar_cols = {}
    for freq in FREQS:
        bar_cols[f'high_{freq}'] = np.zeros(sess_len, dtype=np.int64)
        bar_cols[f'low_{freq}'] = np.zeros(sess_len, dtype=np.int64)
        bar_cols[f'next_ind_{freq}'] = np.zeros(sess_len, dtype=np.int64)

    for freq in FREQS:
        if freq == "day":
            bar_key = np.zeros(sess_len, dtype=np.int64)
            f_ns = 24 * 3600 * 1_000_000_000
        elif freq.endswith("s"):  # second bars, e.g. "1s", "15s"
            f_ns = int(freq[:-1]) * 1_000_000_000
            bar_key = (merged_ts - day0) // f_ns
        else:
            f_ns = int(freq) * 60_000_000_000
            bar_key = (merged_ts - day0) // f_ns

        # Find boundaries of bars
        is_bar_start = np.r_[True, bar_key[1:] != bar_key[:-1]]
        starts = np.flatnonzero(is_bar_start)
        ends = np.r_[starts[1:], sess_len]
        num_bars = len(starts)

        if num_bars == 0:
            resampled_res[freq] = np.zeros(0, dtype=BAR_DTYPE)
            continue

        bar_labels = day0 + bar_key[starts] * f_ns

        # Vectorized aggregation
        bar_high = np.maximum.reduceat(merged_price, starts)
        bar_low = np.minimum.reduceat(merged_price, starts)
        bar_vol = np.add.reduceat(merged_vol, starts)
        bar_open = merged_price[starts]
        bar_close = merged_price[ends - 1]

        # Aggregate news flags (if any tick in the bar is 1, the bar gets 1)
        bar_fomc = np.maximum.reduceat(merged_fomc, starts)
        bar_nfp = np.maximum.reduceat(merged_nfp, starts)
        bar_cpi = np.maximum.reduceat(merged_cpi, starts)
        bar_ppi = np.maximum.reduceat(merged_ppi, starts)
        bar_gdp = np.maximum.reduceat(merged_gdp, starts)

        # Aggregate RTH (if any tick in the bar is in RTH, the bar gets True)
        bar_rth = np.maximum.reduceat(merged_rth, starts)

        # Aggregate session (take the session of the bar's first tick)
        bar_session = merged_session[starts]

        # Aggregate hour (take the hour of the bar's first tick)
        bar_hour = merged_hour[starts]

        # hl calculation
        bar_hl = np.zeros(num_bars, dtype=bool)
        for i in range(num_bars):
            s, e = starts[i], ends[i]
            p = merged_price[s:e]
            h_idx = np.argmax(p)
            l_idx = np.argmin(p)
            if h_idx < l_idx:
                bar_hl[i] = True
            else:
                bar_hl[i] = False



        # Fill the bar columns on each bar's first row (local indices)
        bar_cols[f'high_{freq}'][starts] = bar_high
        bar_cols[f'low_{freq}'][starts] = bar_low
        bar_cols[f'next_ind_{freq}'][starts] = ends # local next_ind

        if freq == "1":
            tick_res.loc[starts, 'start_ind'] = starts # local start_ind

        # Create structured array
        struct_arr = np.zeros(num_bars, dtype=BAR_DTYPE)
        struct_arr['start_ind'] = starts # local start_ind
        struct_arr['ts'] = bar_labels.astype('datetime64[ns]').astype('datetime64[s]')  # labels are ns
        struct_arr['open'] = bar_open
        struct_arr['high'] = bar_high
        struct_arr['low'] = bar_low
        struct_arr['close'] = bar_close
        struct_arr['volume'] = bar_vol
        struct_arr['hl'] = bar_hl
        struct_arr['rth'] = bar_rth # RTH bool from raw data
        struct_arr['session'] = bar_session # 1=Asian, 2=Europe, 3=US
        struct_arr['hour'] = bar_hour
        struct_arr['news_fomc'] = bar_fomc
        struct_arr['news_nfp'] = bar_nfp
        struct_arr['news_cpi'] = bar_cpi
        struct_arr['news_ppi'] = bar_ppi
        struct_arr['news_gdp'] = bar_gdp

        resampled_res[freq] = struct_arr

    for col_name, col_data in bar_cols.items():
        tick_res[col_name] = col_data

    # console.print(f"Session {session_date} processed, rows: {len(session_df)} -> {sess_len}")

    return tick_res, resampled_res, sess_len


def offset_session(tick_res: pd.DataFrame, resampled_res: dict[str, np.ndarray], offset: int) -> np.ndarray:
    """
    One processed session -> its tick.dat rows, with local row numbers moved to global ones by `offset`
    (the number of rows already in tick.dat). Also offsets the bars' start_ind, in place.
    """
    # start_ind is -1 on rows that do not start a 1-min bar, and next_ind is 0 on rows that do not start a bar
    start_mask = tick_res['start_ind'] != -1
    tick_res.loc[start_mask, 'start_ind'] += offset
    tick_res.loc[~start_mask, 'start_ind'] = 0
    for f in FREQS:
        next_col = f'next_ind_{f}'
        tick_res[next_col] = np.where(tick_res[next_col] > 0, tick_res[next_col] + offset, 0)
        if f in PARQUET_FREQS and len(resampled_res[f]) > 0:
            resampled_res[f]['start_ind'] += offset
    # tick.dat is plain int64: only here does ts become seconds on the shifted clock (unix-style)
    tick_res['ts'] = to_unix_epoch(tick_res['ts'])
    return tick_res[DAT_COLS].values.astype(np.int64)


def bars_table(bars: list[np.ndarray]) -> pa.Table:
    """
    Bar arrays from process_session -> one parquet table. ts is the bar's start on the shifted clock
    (New York + 6h), a timestamp in whole seconds like step 1's ts.
    """
    return pa.Table.from_pandas(pd.DataFrame(np.concatenate(bars)), preserve_index=False)


def iter_sessions(src: str, start=None):
    """
    (session_date, ticks DataFrame) for each session of the step-1 file `src`, oldest first. With `start`
    (a date), only ticks from that date on; row groups that end before it are skipped without reading them.
    """
    pf = pq.ParquetFile(src)
    start_ts = pd.Timestamp(start) if start is not None else None
    row_groups = list(range(pf.num_row_groups))
    if start_ts is not None:
        ts_col = pf.schema_arrow.get_field_index('ts')
        stats = [pf.metadata.row_group(i).column(ts_col).statistics for i in row_groups]
        row_groups = [i for i, st in zip(row_groups, stats) if st is None or not st.has_min_max or pd.Timestamp(st.max) >= start_ts]

    current_session = None
    buffer = []
    for batch in pf.iter_batches(row_groups=row_groups):
        df_batch = batch.to_pandas()

        if start_ts is not None:
            if df_batch['ts'].max() < start_ts:
                continue
            df_batch = df_batch[df_batch['ts'] >= start_ts].copy()
            if df_batch.empty:
                continue

        # ts is already New York + 6h from step 1, so its calendar date is the session
        df_batch['session_date'] = df_batch['ts'].dt.date

        for sess_date, group in df_batch.groupby('session_date'):
            if current_session is None:
                current_session = sess_date
            if sess_date != current_session:
                yield current_session, pd.concat(buffer, ignore_index=True)
                buffer = [group]
                current_session = sess_date
            else:
                buffer.append(group)

    if buffer:
        yield current_session, pd.concat(buffer, ignore_index=True)


def check_session(sess_df: pd.DataFrame, sess_date: datetime.date):
    """Every tick of a session must fall inside it: [date 00:00, date 23:00) on the shifted clock."""
    open_ts = pd.Timestamp(sess_date)
    start_time, end_time = sess_df['ts'].min(), sess_df['ts'].max()
    assert open_ts <= start_time and end_time < open_ts + pd.Timedelta(hours=23), f"ticks outside session {sess_date}: {start_time} .. {end_time}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N sessions")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--start", type=str, default=None, help="Skip to date YYYY-MM-DD")
    parser.add_argument("--src", type=str, default=DEFAULT_SRC, help="Step 1 output")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    sessions_processed = 0
    t_start_all = time.monotonic()

    max_workers = min(24, os.cpu_count() or 4)
    executor = ProcessPoolExecutor(max_workers=max_workers)

    in_flight = deque()

    dat_path = os.path.join(args.out, 'tick.dat')
    dat_tmp_path = os.path.join(args.out, 'tick.dat.tmp')

    # We don't know the exact final row count.
    # To append to a memmap, we can open it in 'r+' (or 'w+' initially)
    # but we can't easily dynamically resize a raw memmap cleanly in Windows/Linux without
    # rewriting the file or pre-allocating.
    # A cleaner approach for streaming is a raw binary append.
    # We write to a .tmp file first, and only rename it to .dat if the entire script finishes successfully.
    dat_file = open(dat_tmp_path, 'wb')

    global_offset = 0
    # Bars go to one parquet per freq, written as they come (flushed ~1M rows at a time) so RAM stays flat
    bar_writers = {}
    bar_buffers = {f: [] for f in PARQUET_FREQS}

    def flush_bars(f):
        if not bar_buffers.get(f):
            return
        table = bars_table(bar_buffers[f])
        if f not in bar_writers:
            bar_writers[f] = pq.ParquetWriter(os.path.join(args.out, f'{f}_ohlcv.parquet'), table.schema, compression='zstd')
        bar_writers[f].write_table(table)
        bar_buffers[f] = []

    def write_result(future):
        nonlocal global_offset
        tick_res, resampled_res, sess_len = future.result()
        tick_arr = offset_session(tick_res, resampled_res, global_offset)
        for f in PARQUET_FREQS:
            if len(resampled_res[f]) > 0:
                bar_buffers[f].append(resampled_res[f])
                if sum(len(b) for b in bar_buffers[f]) >= 1_000_000:
                    flush_bars(f)

        # Write the binary bytes directly to the end of the file
        dat_file.write(tick_arr.tobytes())
        global_offset += sess_len

    def submit_session(sess_df, sess_date):
        nonlocal sessions_processed

        print(f"Found session {sess_date}: {len(sess_df)} rows, {sess_df['ts'].min()} to {sess_df['ts'].max()}")
        check_session(sess_df, sess_date)

        future = executor.submit(process_session, sess_df, sess_date)
        in_flight.append(future)

        # Keep at most ~2x max_workers in flight to bound RAM usage
        while len(in_flight) >= max_workers * 2:
            write_result(in_flight.popleft())

        sessions_processed += 1

    for sess_date, sess_df in iter_sessions(args.src, args.start):
        submit_session(sess_df, sess_date)
        if args.limit and sessions_processed >= args.limit:
            break

    # Drain remaining futures
    while in_flight:
        write_result(in_flight.popleft())

    for f in PARQUET_FREQS:
        flush_bars(f)
    for w in bar_writers.values():
        w.close()

    dat_file.close()

    # Atomic rename: guarantees that if tick.dat exists, it is 100% complete.
    os.rename(dat_tmp_path, dat_path)

    t_end_all = time.monotonic()

    print("="*40)
    print(f"Total rows processed: {global_offset}")
    if sessions_processed > 0:
        time_per_session = (t_end_all - t_start_all) / sessions_processed
        print(f"Time per session: {time_per_session:.2f}s")
        print(f"Extrapolated full run (approx 1684 sessions): {time_per_session * 1684 / 60:.2f} mins")

    executor.shutdown()


if __name__ == '__main__':
    main()
