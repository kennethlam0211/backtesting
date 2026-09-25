"""
init / append (data_pipeline.daily_update): appending sessions from MongoDB must give exactly the files a full
build of the same sessions gives. MongoDB is replaced by FakeCollection (ib_ticks.LocalCollection), which
answers the one query ib_ticks makes.
"""
import contextlib
import datetime
import json
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline import daily_update, fake_ib_data, ib_ticks
from data_pipeline import raw_data_preprocessing as step1
from data_pipeline import to_dat as step2
from stop_search.params import DAT_COLS, PARQUET_FREQS

DATES = [datetime.date(2024, 3, d) for d in (4, 5, 6, 7)]  # Mon..Thu, all on ESH4 (instrument 17077)
A, B, C, D = DATES
BAR_FILES = [f"{f}_ohlcv.parquet" for f in PARQUET_FREQS]
ROW_BYTES = len(DAT_COLS) * 8


def raw_session(date, seed):
    """A Databento-like raw file for one session: trades on whole seconds, some seconds with several fills."""
    rng = np.random.default_rng(seed)
    day = pd.Timestamp(date)
    et = pd.date_range(day - pd.Timedelta(hours=6), day + pd.Timedelta(hours=17), freq="1s", tz="America/New_York", inclusive="left")
    et = et[rng.random(len(et)) < 0.15]
    fills = rng.integers(1, 3, len(et))
    price = 5100 + 0.25 * np.cumsum(rng.integers(-1, 2, len(et)))
    n = int(fills.sum())
    return pa.table({
        "ts_event": pa.array(np.repeat(et.tz_convert("UTC").values, fills), pa.timestamp("ns", tz="UTC")),
        "price": pa.array(np.repeat(price, fills), pa.float64()),
        "size": pa.array(rng.integers(1, 20, n), pa.uint32()),
        "instrument_id": pa.array(np.full(n, 17077), pa.uint32()),
    })


RAW = {date: raw_session(date, seed) for seed, date in enumerate(DATES)}


class FakeCollection(ib_ticks.LocalCollection):
    """MongoDB stand-in that also counts the queries."""

    finds = 0

    def find(self, query, projection):
        self.finds += 1
        return super().find(query, projection)


def recorder_docs(dates, time_unit="s"):
    """What the IB recorder would have stored for these sessions: one document per trade, in trade order."""
    docs = []
    for date in dates:
        t = RAW[date].to_pandas()
        secs = t["ts_event"].astype("int64") // 1_000_000_000
        for sec, price, size in zip(secs, t["price"], t["size"]):
            time = {"s": int(sec), "ms": int(sec) * 1000, "datetime": datetime.datetime.fromtimestamp(sec, datetime.timezone.utc).replace(tzinfo=None)}[time_unit]
            docs.append({"_id": len(docs), "time": time, "price": float(price), "size": int(size), "symbol": "ES"})
    # Another instrument in the same collection must be left out
    docs.append({"_id": len(docs), "time": docs[-1]["time"], "price": 18000.0, "size": 1, "symbol": "NQ"})
    return docs


def build(folder, dates):
    """A full build (run init) of `dates` in folder; returns the folder."""
    os.makedirs(folder / "raw_data")
    for date in dates:
        pq.write_table(RAW[date], folder / "raw_data" / f"ES_c_0_trades_{date}.parquet")
    with contextlib.chdir(folder):
        daily_update.init()
    return folder


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """init on all four sessions: what every append below must end up with."""
    return build(tmp_path_factory.mktemp("full"), DATES)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """init on the first two sessions: where the appends start."""
    return build(tmp_path_factory.mktemp("base"), [A, B])


@pytest.fixture
def work(base, tmp_path, monkeypatch):
    """A fresh copy of `base`, as the working directory, so default paths point at it."""
    folder = tmp_path / "work"
    shutil.copytree(base / "data", folder / "data")
    monkeypatch.chdir(folder)
    return folder


def tick_rows(folder):
    return (folder / "data/processed/tick.dat").read_bytes()


def snapshot(folder):
    """Every output file's bytes."""
    return {name: (folder / "data" / name).read_bytes() for name in ["ES_trades_concat.parquet", *(f"processed/{f}" for f in ["tick.dat", *BAR_FILES])]}


def sessions_in(folder):
    """The session dates in each output: step 1's file, tick.dat, and every bar file."""
    ts = pq.read_table(folder / "data/ES_trades_concat.parquet", columns=["ts"]).column("ts").to_pandas()
    tick_ts = np.fromfile(folder / "data/processed/tick.dat", dtype=np.int64).reshape(-1, len(DAT_COLS))[:, DAT_COLS.index("ts")]
    out = {"step1": sorted(set(ts.dt.date)), "tick.dat": sorted(set(pd.to_datetime(tick_ts, unit="s").date))}
    for f in BAR_FILES:
        bar_ts = pq.read_table(folder / "data/processed" / f, columns=["ts"]).column("ts").to_numpy()
        out[f] = sorted(set(pd.to_datetime(bar_ts, unit="s").date))
    return out


def assert_same_as_full(folder, full, n_sessions=len(DATES)):
    """folder's outputs equal the full build's first n_sessions sessions: same step-1 ticks, tick.dat bytes and bars."""
    dates = DATES[:n_sessions]
    ticks = pq.read_table(folder / "data/ES_trades_concat.parquet")
    full_ticks = pq.read_table(full / "data/ES_trades_concat.parquet")
    n = int((full_ticks.column("ts").to_pandas().dt.date <= dates[-1]).sum())
    assert ticks.equals(full_ticks.slice(0, n))

    rows = tick_rows(folder)
    assert rows == tick_rows(full)[:len(rows)]
    assert len(rows) // ROW_BYTES == n  # tick.dat has one row per step-1 tick

    for f in BAR_FILES:
        bars = pq.read_table(folder / "data/processed" / f)
        full_bars = pq.read_table(full / "data/processed" / f)
        keep = pd.to_datetime(full_bars.column("ts").to_numpy(), unit="s").date <= dates[-1]
        assert bars.equals(full_bars.filter(pa.array(keep))), f
    assert not list(folder.glob("data/**/*.tmp"))


# ---------------------------------------------------------------- append == init

def test_append_matches_full_build(work, full):
    fake = FakeCollection(recorder_docs(DATES))
    assert daily_update.append(fake, until=D) is None
    assert_same_as_full(work, full)
    assert fake.finds == 2  # only C and D were read


def test_append_day_by_day_matches_full_build(work, full):
    fake = FakeCollection(recorder_docs(DATES))
    daily_update.append(fake, until=C)
    assert_same_as_full(work, full, n_sessions=3)
    daily_update.append(fake, until=D)
    assert_same_as_full(work, full)


def test_append_again_changes_nothing(work, full):
    fake = FakeCollection(recorder_docs(DATES))
    daily_update.append(fake, until=D)
    before, finds = snapshot(work), fake.finds
    daily_update.append(fake, until=D)
    daily_update.append(fake, until=B)  # older than the data: nothing to add
    assert snapshot(work) == before
    assert fake.finds == finds  # nothing was even read from MongoDB


@pytest.mark.parametrize("time_unit", ["s", "ms", "datetime"])
def test_fetch_time_units(time_unit):
    fake = FakeCollection(recorder_docs([C], time_unit))
    table = ib_ticks.fetch_session(fake, C, time_unit=time_unit)
    raw = RAW[C]
    assert table.column("ts_event").equals(raw.column("ts_event"))
    assert table.column("price").equals(raw.column("price"))
    assert table.column("size").to_pylist() == raw.column("size").to_pylist()  # the NQ trade is left out


def test_fetch_takes_only_the_session():
    fake = FakeCollection(recorder_docs(DATES))
    for date in DATES:
        assert ib_ticks.fetch_session(fake, date).column("ts_event").equals(RAW[date].column("ts_event"))
    assert ib_ticks.fetch_session(fake, datetime.date(2024, 3, 8)).num_rows == 0


# ---------------------------------------------------------------- closures, missing data

def test_session_without_ticks_stops_the_append(work):
    fake = FakeCollection(recorder_docs([A, B, D]))  # the recorder missed C
    before = snapshot(work)
    assert daily_update.append(fake, until=D) == C
    assert snapshot(work) == before  # D is not appended past the gap


def test_holiday_without_ticks_is_skipped(work, monkeypatch):
    monkeypatch.setattr(ib_ticks, "HOLIDAYS", {C})
    fake = FakeCollection(recorder_docs([A, B, D]))
    assert daily_update.append(fake, until=D) is None
    assert sessions_in(work) == {k: [A, B, D] for k in ["step1", "tick.dat", *BAR_FILES]}


def test_holiday_that_traded_is_appended(work, full, monkeypatch):
    monkeypatch.setattr(ib_ticks, "HOLIDAYS", {C})  # e.g. a shortened session listed as a holiday
    assert daily_update.append(FakeCollection(recorder_docs(DATES)), until=D) is None
    assert_same_as_full(work, full)


def test_holidays_yaml_formats(tmp_path):
    listed = tmp_path / "list.yaml"
    listed.write_text("- 2026-12-25\n- '2027-01-01'\n")
    named = tmp_path / "named.yaml"
    named.write_text("2026-12-25: Christmas\n2027-01-01: New Year\n")
    empty = tmp_path / "empty.yaml"
    empty.write_text("# nothing yet\n")
    expected = {datetime.date(2026, 12, 25), datetime.date(2027, 1, 1)}
    assert ib_ticks.load_holidays(listed) == ib_ticks.load_holidays(named) == expected
    assert ib_ticks.load_holidays(empty) == set()
    assert ib_ticks.HOLIDAYS == ib_ticks.load_holidays()  # the placeholder in params/ loads


def test_main_exit_codes(work, monkeypatch, capsys):
    fake = FakeCollection(recorder_docs([A, B, C]))
    monkeypatch.setattr(ib_ticks, "connect", lambda: fake)
    assert daily_update.main(["append", "--until", str(D)]) == 1  # C appended, then D has no ticks
    assert "no ticks for session 2024-03-07" in capsys.readouterr().err
    assert sessions_in(work)["tick.dat"] == [A, B, C]
    assert daily_update.main(["append", "--until", str(C)]) == 0  # up to date


def test_open_session_is_refused(work, monkeypatch, capsys):
    monkeypatch.setattr(ib_ticks, "connect", lambda: pytest.fail("MongoDB must not be read"))
    today = ib_ticks.last_closed_session() + datetime.timedelta(days=1)
    assert daily_update.main(["append", "--until", str(today)]) == 1
    assert "has not closed yet" in capsys.readouterr().err


def test_bad_until_is_a_usage_error(work):
    with pytest.raises(SystemExit) as e:
        daily_update.main(["append", "--until", "2026-13-01"])
    assert e.value.code == 2


def test_one_run_at_a_time(work):
    fcntl = pytest.importorskip("fcntl")
    with open("data/ES_trades_concat.parquet.lock", "w") as held:  # as init or another append holds it
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another init or append"):
            daily_update.append(FakeCollection(recorder_docs(DATES)), until=D)


def test_prices_rounded_to_the_tick_grid():
    docs = [{"time": 1709679600, "price": 5100.2499999, "size": 1, "symbol": "ES"}]
    assert ib_ticks.fetch_session(FakeCollection(docs), C).column("price").to_pylist() == [5100.25]
    docs[0]["price"] = 5100.1
    with pytest.raises(ValueError, match="off the 0.25 grid"):
        ib_ticks.fetch_session(FakeCollection(docs), C)


def test_default_paths_agree():
    assert step2.DEFAULT_SRC == step1.DEFAULT_OUT  # init (step 2's default) and append read the same file


def test_main_without_data(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ib_ticks, "connect", lambda: FakeCollection([]))
    assert daily_update.main(["append", "--until", str(C)]) == 1
    assert "daily_update init" in capsys.readouterr().err


# ---------------------------------------------------------------- failures leave the files usable

def test_mongo_error_leaves_files_untouched(work):
    fake = FakeCollection(recorder_docs(DATES))
    real_find = fake.find
    calls = []

    def flaky_find(query, projection):
        calls.append(1)
        if len(calls) == 2:
            raise ConnectionError("MongoDB went away")
        return real_find(query, projection)

    fake.find = flaky_find
    before = snapshot(work)
    with pytest.raises(ConnectionError):
        daily_update.append(fake, until=D)  # C is read, D fails: nothing is written
    assert snapshot(work) == before
    assert not list(work.glob("data/**/*.tmp"))


def test_step2_failure_is_caught_up_by_the_next_run(work, full, monkeypatch):
    fake = FakeCollection(recorder_docs(DATES))
    before = snapshot(work)

    def disk_full(bars):
        raise OSError("no space left on device")

    with monkeypatch.context() as m:
        m.setattr(step2, "bars_table", disk_full)  # after the rows went onto tick.dat
        with pytest.raises(OSError):
            daily_update.append(fake, until=D)
    # Step 1 has C and D; tick.dat and the bars are as they were
    assert sessions_in(work)["step1"] == DATES
    assert all(snapshot(work)[f"processed/{f}"] == before[f"processed/{f}"] for f in ["tick.dat", *BAR_FILES])
    assert not list(work.glob("data/**/*.tmp"))

    finds = fake.finds
    daily_update.append(fake, until=D)
    assert fake.finds == finds  # step 2 catches up from step 1's file, not from MongoDB
    assert_same_as_full(work, full)


def test_cut_off_append_is_detected(work):
    # A killed append (power cut, OOM) can leave rows in tick.dat that the bar files do not cover
    with open(work / "data/processed/tick.dat", "ab") as f:
        f.write(bytes(ROW_BYTES * 3))
    with pytest.raises(ValueError, match="python -m data_pipeline.to_dat"):
        step2.append_sessions()
    assert step2.committed_rows("data/processed") < os.path.getsize("data/processed/tick.dat") // ROW_BYTES


def test_fake_ib_data_appends_from_file(work, tmp_path, capsys):
    jsonl = tmp_path / "fake.jsonl"
    fake_ib_data.main(["--days", "2", "--out", str(jsonl)])
    lines = jsonl.read_text().splitlines()
    assert set(json.loads(lines[0])) == set(ib_ticks.FIELDS.values())
    assert daily_update.main(["append", "--file", str(jsonl), "--until", str(D)]) == 0
    assert sessions_in(work) == {k: DATES for k in ["step1", "tick.dat", *BAR_FILES]}  # the 2 weekdays after B
    rows = os.path.getsize("data/processed/tick.dat") // ROW_BYTES
    assert step2.committed_rows("data/processed") == rows
    assert rows == pq.ParquetFile("data/ES_trades_concat.parquet").metadata.num_rows
    # The fake prices carry on from the last real one
    ticks = pq.read_table("data/ES_trades_concat.parquet", columns=["ts", "price"]).to_pandas()
    last_real = ticks[ticks.ts.dt.date == B].price.iloc[-1]
    first_fake = ticks[ticks.ts.dt.date == C].price.iloc[0]
    assert abs(int(first_fake) - int(last_real)) <= 1


def test_append_to_other_paths(work, full, tmp_path):
    folder = tmp_path / "copy"
    shutil.copytree(work / "data", folder)
    before = snapshot(work)
    jsonl = tmp_path / "docs.jsonl"
    jsonl.write_text("".join(json.dumps({k: v for k, v in doc.items() if k != "_id"}) + "\n" for doc in recorder_docs(DATES)))
    argv = ["append", "--file", str(jsonl), "--until", str(D), "--src", str(folder / "ES_trades_concat.parquet"), "--out", str(folder / "processed")]
    assert daily_update.main(argv) == 0
    assert snapshot(work) == before  # the default outputs are untouched
    assert (folder / "processed/tick.dat").read_bytes() == tick_rows(full)


def test_step1_refuses_older_session(work):
    day = step1.to_ticks(RAW[A], pd.Timestamp(A), "ESH4")
    with pytest.raises(ValueError, match="not newer"):
        step1.append_sessions(step1.DEFAULT_OUT, [day])
    assert not list(work.glob("data/*.tmp"))


# ---------------------------------------------------------------- session calendar

@pytest.mark.parametrize("now, session", [
    ("2026-09-25 17:00", "2026-09-25"),  # Friday, at the close
    ("2026-09-25 16:59", "2026-09-24"),  # Friday, still trading
    ("2026-09-26 12:00", "2026-09-25"),  # Saturday
    ("2026-09-27 19:00", "2026-09-25"),  # Sunday evening: Monday's session is open, not closed
    ("2026-09-28 09:00", "2026-09-25"),  # Monday morning
    ("2026-09-29 05:00", "2026-09-28"),  # Tuesday before the open
])
def test_last_closed_session(now, session):
    assert ib_ticks.last_closed_session(pd.Timestamp(now, tz="America/New_York")) == datetime.date.fromisoformat(session)


def test_last_closed_session_from_another_time_zone():
    # 06:30 Tuesday in Hong Kong is 18:30 Monday in New York (EDT): Monday's session has closed
    assert ib_ticks.last_closed_session(pd.Timestamp("2026-09-29 06:30", tz="Asia/Hong_Kong")) == datetime.date(2026, 9, 28)


@pytest.mark.parametrize("session, start, end", [
    ("2024-03-08", "2024-03-07 23:00", "2024-03-08 22:00"),  # EST
    ("2024-03-11", "2024-03-10 22:00", "2024-03-11 21:00"),  # first Monday of EDT (clocks changed Sunday 02:00)
    ("2024-11-04", "2024-11-03 23:00", "2024-11-04 22:00"),  # first Monday of EST again
])
def test_session_window_utc(session, start, end):
    assert ib_ticks.session_window(session) == (pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))


def test_session_dates_skip_weekends():
    assert ib_ticks.session_dates("2026-09-24", "2026-09-29") == [datetime.date(2026, 9, d) for d in (25, 28, 29)]
    assert ib_ticks.session_dates("2026-09-25", "2026-09-25") == []
