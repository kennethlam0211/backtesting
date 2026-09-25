import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rich.traceback
import yaml
from rich import print

rich.traceback.install()

# Globex reopens 18:00 New York time; +6h puts that at 00:00 and the close (17:00 ET) at 23:00,
# so every session is exactly one calendar date: [date 00:00, date 23:00).
SHIFT = pd.Timedelta(hours=6)

# Roll days: the main file opens on the OLD contract and switches at 00:00 UTC. Each block holds the
# NEW contract for that gap [18:00 ET, 00:00 UTC), bought as ES.c.1.
BLOCK_DIR = Path("raw_data/roll_open_blocks")

# Found from the repo root, whatever the working directory: the news calendar lives in params/,
# the contract-code table next to the raw Databento files in raw_data/
REPO_DIR = Path(__file__).resolve().parents[1]
PARAMS_DIR = REPO_DIR / "params"

# instrument_id -> contract code (ESH0 .. ESZ6). Databento numbers can be reused over the years,
# so a match needs both the number and the session date range.
PDT_CODES = pd.read_csv(REPO_DIR / "raw_data" / "pdt_codes.csv")


# News events window: 5 mins before to 5 mins after
NEWS_WINDOW = pd.Timedelta(minutes=5)

def load_news_events():
    yaml_path = PARAMS_DIR / "news_events.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"News events file not found at {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    events = {}
    for ev_type in ['fomc', 'nfp', 'cpi', 'ppi', 'gdp']:
        if ev_type not in data:
            continue

        time_et = data[ev_type]['time_et']
        # Default all dates to the standard time
        dt_strings = pd.Series(data[ev_type]['dates']) + ' ' + str(time_et)

        # Apply specific date time overrides if they exist
        if 'times' in data[ev_type]:
            overrides = data[ev_type]['times']
            for o_date, o_time in overrides.items():
                if o_date in data[ev_type]['dates']:
                    idx = data[ev_type]['dates'].index(o_date)
                    dt_strings.iloc[idx] = o_date + ' ' + str(o_time)

        # Convert to datetime and apply shift
        ts = pd.to_datetime(dt_strings) + SHIFT

        # Store as sorted array of windows (start, end)
        starts = (ts - NEWS_WINDOW).values
        ends = (ts + NEWS_WINDOW).values

        # Sort just in case yaml wasn't
        sort_idx = np.argsort(starts)
        events[ev_type] = (starts[sort_idx], ends[sort_idx])

    return events

# No fallback: a missing yaml must stop the run, not silently write all-zero news flags
NEWS_EVENTS = load_news_events()



def pdt_code_for(instrument_id, date):
    m = PDT_CODES[(PDT_CODES.instrument_id == instrument_id) & (PDT_CODES.first_session <= date) & (date <= PDT_CODES.last_session)]
    assert len(m) == 1, f"no contract code for instrument_id {instrument_id} on {date}"
    return m.pdt_code.iloc[0]


def splice_roll(table, block_file):
    """Roll day -> one contract from the open: the block, then the main file's new-contract rows."""
    block = pq.read_table(block_file).replace_schema_metadata(None)
    assert block.schema.equals(table.schema), f"block schema differs: {block_file.name}"
    new_ids = pc.unique(block.column('instrument_id')).to_pylist()
    assert len(new_ids) == 1, f"block holds several contracts: {block_file.name}"
    new_rows = table.filter(pc.equal(table.column('instrument_id'), new_ids[0]))
    # The block must end before the main file's new-contract rows start: no overlap, time order kept
    assert new_rows.num_rows and pc.max(block.column('ts_event')).value < pc.min(new_rows.column('ts_event')).value, f"block overlaps main: {block_file.name}"
    return pa.concat_tables([block, new_rows])


def apply_news_flags(out, session_date):
    """
    Applies news flags in-place to the output DataFrame.
    """
    for ev_type in ['fomc', 'nfp', 'cpi', 'ppi', 'gdp']:
        flag_col = f'news_{ev_type}'
        out[flag_col] = 0
        if ev_type in NEWS_EVENTS:
            starts, ends = NEWS_EVENTS[ev_type]
            # Find windows that overlap this session
            s_end = session_date + pd.Timedelta(hours=23)
            # Window overlaps session if: start < session_end AND end > session_start
            # Using binary search since arrays are sorted
            idx_start = np.searchsorted(ends, session_date, side='right')
            idx_end = np.searchsorted(starts, s_end, side='left')

            ts_vals = out['ts'].values

            for i in range(idx_start, idx_end):
                w_start = starts[i]
                w_end = ends[i]

                # ticks inside [w_start, w_end)
                tick_start = np.searchsorted(ts_vals, w_start, side='left')
                tick_end = np.searchsorted(ts_vals, w_end, side='left')

                if tick_end > tick_start:
                    out.iloc[tick_start:tick_end, out.columns.get_loc(flag_col)] = 1

        # Cast to int8 for memory efficiency
        out[flag_col] = out[flag_col].astype('int8')

def to_ticks(table, session_date, pdt_code):
    """One raw single-contract session -> merged ticks with columns ts, price, volume, rth, session,
    hour and the five news flags.

    ts = ts_event (UTC) as New York wall clock + 6h, tz-naive, floored to the second after the
    merge. price is in 0.25-point ticks (x4). Rows with the same ts (ns) and price are merged,
    summing size into volume, in first-appearance order. Identical raw rows are separate fills,
    so summing (never dropping) keeps the volume exact. pdt_code is the session's contract,
    e.g. ESM4; the caller has already checked it, and it is not written.
    """
    df = table.select(['ts_event', 'price', 'size']).to_pandas()
    # Wall clock first (handles daylight saving), then add the shift as plain clock arithmetic
    ts = df['ts_event'].dt.tz_convert("America/New_York").dt.tz_localize(None) + SHIFT
    # Every tick must fall inside its file's session: [date 00:00, date 23:00)
    since_open = ts - session_date
    assert ((since_open >= pd.Timedelta(0)) & (since_open < pd.Timedelta(hours=23))).all(), f"tick outside session {session_date.date()}"

    df = pd.DataFrame({'ts': ts, 'price': df['price'], 'size': df['size'].astype('int64')})
    # ES trades in 0.25-point ticks; price x4 is the price in ticks, an exact integer (4000.25 -> 16001).
    # OHLC downstream is built from it, so every price column is in ticks. int32 holds any ES price
    # (int16 would wrap above 32,767 ticks = 8,191.75 points); to_dat.py widens it to int64 for tick.dat.
    df['price'] = (df['price'] * 4).astype('int32')

    out = df.groupby(['ts', 'price'], sort=False).agg(volume=('size', 'sum')).reset_index()
    out['pdt_code'] = pdt_code
    assert out['volume'].sum() == df['size'].sum(), f"volume changed in merge {session_date.date()}"
    # Seconds are enough from here on: the merge used full ns, and row order keeps the trade order.
    # Calculate Regular Trading Hours (RTH) based on shifted time
    # RTH is 09:30 - 16:00 ET. In our +6h shifted clock, that's 15:30 to 22:00.
    rth_start = pd.Timedelta(hours=15, minutes=30)
    rth_end = pd.Timedelta(hours=22)
    time_since_midnight = out['ts'] - out['ts'].dt.normalize()
    out['rth'] = ((time_since_midnight >= rth_start) & (time_since_midnight < rth_end)).astype('int8')

    # Add 8-hour sessions: 1 (Asian), 2 (Europe), 3 (US)
    out['session'] = ((time_since_midnight.dt.components.hours // 8) + 1).astype('int8')

    # Add hour column directly from the shifted ts
    out['hour'] = out['ts'].dt.hour.astype('int8')

    # Floor to whole seconds, kept as a timestamp (step 2 turns it into integer seconds for tick.dat)
    out['ts'] = out['ts'].dt.floor('s').astype('datetime64[s]')

    # Add news flags
    apply_news_flags(out, session_date)

    cols = ['ts', 'price', 'volume', 'rth', 'session', 'hour', 'news_fomc', 'news_nfp', 'news_cpi', 'news_ppi', 'news_gdp']
    return pa.Table.from_pandas(out[cols], preserve_index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="0000-00-00", help="first session date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default="9999-99-99", help="last session date YYYY-MM-DD (inclusive)")
    parser.add_argument("--out", default="data/ES_trades_concat.parquet")
    args = parser.parse_args()

    raw_dir = Path("raw_data")
    out_file = Path(args.out)

    # Get all files matching pattern, excluding "_partial".
    # Sort by the session date, not the full name: ES_c_0_ and ES_v_0_ dates interleave.
    all_files = sorted(raw_dir.glob("ES_*_trades_*.parquet"), key=lambda f: f.stem.split("_trades_")[1])
    files = [f for f in all_files if args.start <= f.stem.split("_trades_")[1] <= args.end and not f.name.endswith("_partial.parquet")]

    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} files to concatenate.")

    # Every raw file must have the first file's schema
    base_schema = pq.read_schema(files[0]).remove_metadata()

    writer = None
    raw_rows = 0
    spliced = 0
    total_rows = 0
    first_ts = None
    last_ts = None

    seen_dates = set()

    try:
        for i, f in enumerate(files, 1):
            date = f.stem.split("_trades_")[1]
            assert date not in seen_dates, f"Duplicate session date {date} in {f.name}"
            seen_dates.add(date)

            table = pq.read_table(f)
            # Fail on any real schema difference instead of silently casting it away
            assert table.schema.remove_metadata().equals(base_schema), f"schema differs: {f.name}"
            raw_rows += table.num_rows
            date = f.stem.split("_trades_")[1]
            table = table.replace_schema_metadata(None)
            block = BLOCK_DIR / f"ES_c_1_open_block_{date}.parquet"
            if block.exists():
                table = splice_roll(table, block)
                spliced += 1
            # Every session must now be a single contract (a roll day without a block would fail here)
            ids = pc.unique(table.column('instrument_id')).to_pylist()
            assert len(ids) == 1, f"several contracts in {f.name}"
            # pdt_code_for(ids[0], date) will still assert it matches exactly one known contract code,
            # but we drop it inside to_ticks so it doesn't take up disk space.
            table = to_ticks(table, pd.Timestamp(date), pdt_code_for(ids[0], date))

            if writer is None:
                writer = pq.ParquetWriter(out_file, table.schema, compression='zstd')

            # Track timestamps
            if table.num_rows > 0:
                ts_col = table.column('ts')
                if first_ts is None:
                    first_ts = ts_col[0].as_py()
                last_ts = ts_col[table.num_rows - 1].as_py()

            writer.write_table(table)
            total_rows += table.num_rows

            if i % 100 == 0:
                print(f"Processed {i}/{len(files)} files... (raw rows {raw_rows:,} -> merged {total_rows:,})")

    finally:
        if writer:
            writer.close()

    out_size = out_file.stat().st_size
    print("\nConcat Summary:")
    print(f"Files processed: {len(files)}")
    print(f"Roll days spliced with an open block: {spliced}")
    print(f"Raw rows: {raw_rows:,} -> merged ticks: {total_rows:,}")
    print(f"First ts: {first_ts}")
    print(f"Last ts: {last_ts}")
    print(f"Output file size: {out_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
