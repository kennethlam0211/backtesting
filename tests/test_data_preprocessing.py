"""
data_pipeline/data_preprocessing.py (polars) against reference/preprocessing_pandas.py, and no look-ahead:
- every higher-freq bar appears on the 1-min row during which it closes, not sooner and not later;
- cutting the data off at any time T leaves every row that ended by T exactly as it was (nothing reads ahead).
The bars come from the real pipeline (to_ticks -> process_session), with many minutes without trades.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import reference.preprocessing_pandas as ref
from data_pipeline import data_preprocessing as dp
from data_pipeline import raw_data_preprocessing as step1
from data_pipeline import to_dat as step2
from params import UD_PIVOTS
from stop_search.params import PARQUET_FREQS

SESSIONS = ["2024-03-04", "2024-03-05", "2024-03-06"]
FREQS = ["1", "15", "60", "day"]


def raw_session(date, rng):
    """Sparse raw trades (about 30% of minutes have none, as overnight), busy in the session's last 2 minutes."""
    day = pd.Timestamp(date)
    secs = pd.date_range(day - pd.Timedelta(hours=6), day + pd.Timedelta(hours=17), freq="1s", tz="America/New_York", inclusive="left")
    busy = secs >= (day + pd.Timedelta(hours=16, minutes=58)).tz_localize("America/New_York")
    secs = secs[rng.random(len(secs)) < np.where(busy, 0.5, 0.02)]
    price = 5100 + 0.25 * np.cumsum(rng.integers(-2, 3, len(secs)))
    return pa.table({
        "ts_event": pa.array(secs.tz_convert("UTC").values, pa.timestamp("ns", tz="UTC")),
        "price": pa.array(price, pa.float64()),
        "size": pa.array(rng.integers(1, 10, len(secs)), pa.int64()),
    })


RNG = np.random.default_rng(5)
TICKS = {date: step1.to_ticks(raw_session(date, RNG), pd.Timestamp(date), "ESH4").to_pandas() for date in SESSIONS}


def write_bars(folder, cutoff=None):
    """The bar files of the three sessions as step 2 writes them; with `cutoff`, only the ticks before it."""
    bars = {f: [] for f in PARQUET_FREQS}
    for date, ticks in TICKS.items():
        if cutoff is not None:
            ticks = ticks[ticks["ts"] < cutoff]
        if len(ticks):
            _, resampled, _ = step2.process_session(ticks.reset_index(drop=True), pd.Timestamp(date).date())
            for f in PARQUET_FREQS:
                bars[f].append(resampled[f])
    for f in PARQUET_FREQS:
        pq.write_table(step2.bars_table(bars[f]), folder / f"{f}_ohlcv.parquet")
    return folder


def build(folder, freqs=FREQS, keep_warmup=True):
    """The merged table; the warm-up is kept by default (3 sessions are too few for 5 day pivots)."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(dp, "FREQS", freqs)
        return dp.DataPreprocessor(str(folder)).build_merged_dataset(keep_warmup=keep_warmup).to_pandas()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return write_bars(tmp_path_factory.mktemp("bars"))


@pytest.fixture(scope="module")
def merged(data_dir):
    return build(data_dir)


def test_features_match_the_pandas_reference(data_dir):
    new = dp.DataPreprocessor(str(data_dir)).process_frequency("1").to_pandas()
    old = pd.read_parquet(data_dir / "1_ohlcv.parquet").sort_values("ts").reset_index(drop=True)
    ref.sma(old, 20)
    ref.std(old, 20)
    ref.atr(old, 20)
    ref.sma_rsi(old, 20)
    old = ref.fvg(old, 20)
    old["20_std_1"] = old["20_std"]
    old = ref.UD_cal(old, "close", 20, "1")

    for col in ["20_sma", "20_std", "20_atr", "20_sma_rsi", "20_fvg"]:
        np.testing.assert_allclose(new[f"{col}_1"].to_numpy(float), old[col].to_numpy(float),
                                   rtol=0, atol=1e-6, equal_nan=True, err_msg=col)
    # U/D: the reference picks its starting side from 10 bars ahead; without that peek the levels differ
    # until the state first resets (about 35 bars at most), then match exactly
    for col in ["20_U", "20_D", "20_UD_flag"]:
        np.testing.assert_allclose(new[f"{col}_1"].to_numpy(float)[100:], old[f"{col}_1"].to_numpy(float)[100:],
                                   rtol=0, atol=1e-6, equal_nan=True, err_msg=col)
    assert (old["20_U_1"].iloc[100:] > 0).any()  # the U/D levels were really compared


def reference_live_std(merged, bars, freq):
    """
    The reference's unit_std(raw_20 + [close_1]) at every 1-min row: raw_20 holds the last 21 closes of the
    bars shown so far, and unit_std keeps the last 20 of the list, so 19 closes plus the current 1-min close.
    Unlike the reference, NaN until 19 bars have closed (no std of a partial window).
    """
    length = 24 * 3600 if freq == "day" else int(freq) * 60
    closes = np.minimum(bars["ts"] + length, bars["ts"] // 86400 * 86400 + 23 * 3600).to_numpy()
    shown = np.searchsorted(closes, merged["ts"].to_numpy() + 60, side="right") - 1
    px = bars["close"].to_numpy(float)
    return np.array([np.std((list(px[max(0, k - 20):k + 1]) + [c])[-20:]) if k >= 18 else np.nan
                     for k, c in zip(shown, merged["close_1"].to_numpy(float))])


@pytest.mark.parametrize("freq", ["15", "60"])
def test_higher_freq_ud_follows_the_reference(data_dir, merged, freq):
    bars = pd.read_parquet(data_dir / f"{freq}_ohlcv.parquet").sort_values("ts").reset_index(drop=True)
    std = reference_live_std(merged, bars, freq)
    np.testing.assert_allclose(merged[f"20_std_live_{freq}"].to_numpy(float), std, rtol=1e-9, atol=1e-6, equal_nan=True)

    # UD(df, 20, freq) = UD_cal on close_1 with that std; equal once the start has passed (see the U/D note above)
    old = ref.UD_cal(pd.DataFrame({"close_1": merged["close_1"], f"20_std_{freq}": std}), "close_1", 20, freq)
    start = int(np.argmax(~np.isnan(std))) + 300  # once the std exists and the start has passed
    for col in [f"20_U_{freq}", f"20_D_{freq}", f"20_UD_flag_{freq}"]:
        np.testing.assert_allclose(merged[col].to_numpy(float)[start:], old[col].to_numpy(float)[start:],
                                   rtol=0, atol=1e-6, equal_nan=True, err_msg=col)
    assert merged[f"20_U_{freq}"].iloc[start:].notna().sum() > 3  # real pivots were compared


@pytest.mark.parametrize("freq", ["1", "15", "60"])
def test_last_pivots(merged, freq):
    u, d = merged[f"20_U_{freq}"].to_numpy(float), merged[f"20_D_{freq}"].to_numpy(float)
    seen = []
    for i in range(len(merged)):
        seen += [x for x in (d[i], u[i]) if not np.isnan(x)]
        want = (seen[::-1] + [np.nan] * UD_PIVOTS)[:UD_PIVOTS]
        got = [merged[f"20_UD_last{k + 1}_{freq}"].iloc[i] for k in range(UD_PIVOTS)]
        np.testing.assert_array_equal(got, want, err_msg=f"row {i}")
    assert len(seen) > UD_PIVOTS


def test_bollinger_bands_are_2_sigma(data_dir):
    new = dp.DataPreprocessor(str(data_dir)).process_frequency("1").to_pandas()
    sma, std = new["20_sma_1"], new["20_std_1"]
    ok = std > 0
    np.testing.assert_allclose(new["20_bbands_1"][ok], ((new["close_1"] - sma) / (2 * std))[ok])
    np.testing.assert_allclose(new["20_bband_upper_1"], sma + 2 * std)
    np.testing.assert_allclose(new["20_bband_lower_1"], sma - 2 * std)


@pytest.mark.parametrize("freq", ["15", "60", "day"])
def test_higher_freq_bar_appears_when_it_closes(data_dir, merged, freq):
    bars = pd.read_parquet(data_dir / f"{freq}_ohlcv.parquet").sort_values("ts").reset_index(drop=True)
    length = 24 * 3600 if freq == "day" else int(freq) * 60
    closes = np.minimum(bars["ts"] + length, bars["ts"] // 86400 * 86400 + 23 * 3600)  # sessions end at 23:00

    rows = merged["ts"].to_numpy()
    # The newest bar that has closed by the end of each 1-min row (row ts + 60 s)
    k = np.searchsorted(closes.to_numpy(), rows + 60, side="right") - 1
    expected = np.where(k >= 0, bars["close"].to_numpy()[np.maximum(k, 0)], np.nan)
    got = merged[f"close_{freq}"].to_numpy(float)
    np.testing.assert_array_equal(got, expected)

    if freq != "day":  # and the join really was tested across gaps: some bars close in a minute without trades
        assert not np.isin(closes - 60, rows).all()


def test_day_bar_waits_for_the_session_end(merged):
    ts = pd.to_datetime(merged["ts"], unit="s")
    first_day = ts.dt.date == pd.Timestamp(SESSIONS[0]).date()
    before_close = first_day & (ts.dt.time < pd.Timestamp("22:59").time())
    assert merged.loc[before_close, "close_day"].isna().all()  # no day bar before its session closes
    second_day = ts.dt.date == pd.Timestamp(SESSIONS[1]).date()
    assert merged.loc[second_day, "close_day"].notna().all()  # the previous session's bar all day


def cutoffs():
    """
    Cut times inside the last minute of 15-min, 60-min and day bars, with trades of that bar still to come:
    where a bar shown even a minute early would be missing them. Plus a few times anywhere.
    """
    ts = TICKS[SESSIONS[1]]["ts"]
    sec = ((ts - ts.dt.normalize()).dt.total_seconds()).astype(int).to_numpy()
    out = []
    for minutes in (15, 60, 23 * 60):  # the day bar ends with the session at 23:00
        length = minutes * 60
        last_minute = sec % length >= length - 60
        found = 0
        for bar in np.unique(sec[last_minute] // length):
            times = np.unique(sec[last_minute & (sec // length == bar)])
            if len(times) >= 2 and found < 2:
                out.append(ts.dt.normalize().iloc[0] + pd.Timedelta(seconds=int(times[0]) + 1))
                found += 1
    return out + [pd.Timestamp(t) for t in ("2024-03-05 03:17:31", "2024-03-06 15:45:00")]


def test_no_look_ahead_cut_the_data_at_any_time(merged, tmp_path):
    # Everything is rebuilt from the ticks before each cutoff: the rows that ended by then must not change
    cuts = cutoffs()
    assert len(cuts) >= 6
    for i, cut in enumerate(cuts):
        folder = tmp_path / str(i)
        folder.mkdir()
        part = build(write_bars(folder, cutoff=cut))
        done = lambda df: df[df["ts"] + 60 <= int(cut.timestamp())].reset_index(drop=True)
        pd.testing.assert_frame_equal(done(part), done(merged), obj=f"rows ended by {cut}")


def test_ud_levels_never_read_ahead():
    rng = np.random.default_rng(9)
    for trial in range(20):
        px = 20400 + np.cumsum(rng.integers(-3, 4, 2000)).astype(float)
        if trial % 2:
            px[:11] = px[0] + np.arange(11)  # rising at the start: where the reference peeked 10 bars ahead
        kv = pd.Series(px).rolling(20).std(ddof=0).to_numpy()
        full = dp._calc_ud_levels(px, kv)
        for k in (1, 5, 11, 25, 500):
            for a, b in zip(dp._calc_ud_levels(px[:k], kv[:k]), full):
                np.testing.assert_array_equal(a, b[:k])


def test_live_std_needs_19_closed_bars(merged, data_dir):
    bars = pd.read_parquet(data_dir / "60_ohlcv.parquet").sort_values("ts").reset_index(drop=True)
    closes = np.minimum(bars["ts"] + 3600, bars["ts"] // 86400 * 86400 + 23 * 3600).to_numpy()
    shown = np.searchsorted(closes, merged["ts"].to_numpy() + 60, side="right")  # bars closed by each row
    std = merged["20_std_live_60"].to_numpy(float)
    assert np.isnan(std[shown < 19]).all() and not np.isnan(std[shown >= 19]).any()
    assert np.isnan(merged["20_U_60"].to_numpy(float)[shown < 19]).all()  # no pivot on a partial window


def test_warmup_is_dropped(data_dir):
    freqs = ["1", "15", "60"]  # 3 sessions are enough for these to get their pivots, not for the day
    full = build(data_dir, freqs)
    trimmed = build(data_dir, freqs, keep_warmup=False)
    pivots = full[[f"20_UD_last{UD_PIVOTS}_{f}" for f in freqs]].notna().all(axis=1)
    first = int(pivots.to_numpy().argmax())
    assert 0 < first < len(full) and pivots.iloc[first:].all() and not pivots.iloc[first - 1]
    # (dtypes may differ: pandas turns an int column with a null in the warm-up into float)
    pd.testing.assert_frame_equal(trimmed, full.iloc[first:].reset_index(drop=True), check_dtype=False)
    with pytest.raises(ValueError, match="more data is needed"):
        build(data_dir, FREQS, keep_warmup=False)  # the day never gets 5 pivots in 3 sessions


def test_main_writes_next_to_the_bars(data_dir, capsys, monkeypatch):
    monkeypatch.setattr(dp, "FREQS", ["1", "15", "60"])
    dp.main(["--data-dir", str(data_dir)])
    out = pd.read_parquet(data_dir / "training_data.parquet")
    assert 0 < len(out) < len(pd.read_parquet(data_dir / "1_ohlcv.parquet"))  # the warm-up is dropped
    assert "Warm-up dropped" in capsys.readouterr().out
    dp.main(["--data-dir", str(data_dir), "--keep-warmup"])
    assert len(pd.read_parquet(data_dir / "training_data.parquet")) == len(pd.read_parquet(data_dir / "1_ohlcv.parquet"))
