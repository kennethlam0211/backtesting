import os
import sys
import datetime
import subprocess

import numpy as np
import pandas as pd
import pytest

from data_pipeline.to_dat import offset_session, process_session
from stop_search import StopSearch, CHILD, DAT_COLS, DAT_DTYPE, FREQS

PRICE_COL = DAT_COLS.index('price')

NEXT = {f: DAT_COLS.index(f'next_ind_{f}') for f in FREQS}
HIGH = {f: DAT_COLS.index(f'high_{f}') for f in FREQS}
LOW = {f: DAT_COLS.index(f'low_{f}') for f in FREQS}


def make_session(rng, date):
    """One session of step-1 ticks: bursts and quiet gaps, several ticks per second, price in ticks (x4)."""
    secs = np.cumsum(rng.exponential(rng.choice([0.1, 1.0, 20.0], size=40_000)))
    secs = secs[secs < 23 * 3600].astype(np.int64)
    n = len(secs)
    df = pd.DataFrame({
        'ts': pd.Timestamp(date) + pd.to_timedelta(secs, unit='s'),
        'price': 20_000 + np.cumsum(rng.integers(-1, 2, n)),
        'volume': rng.integers(1, 5, n),
    })
    df['ts'] = df['ts'].astype('datetime64[s]')
    for c in ['rth', 'session', 'hour', 'news_fomc', 'news_nfp', 'news_cpi', 'news_ppi', 'news_gdp']:
        df[c] = np.int8(0)
    return df


def table(rows):
    """tick.dat rows (DAT_DTYPE) as an int64 table, columns in DAT_COLS order: what the references below read."""
    return np.stack([rows[c].astype(np.int64) for c in DAT_COLS], axis=1)


def pack(tab):
    """An int64 table (DAT_COLS order) back to tick.dat rows (DAT_DTYPE): what StopSearch reads."""
    rows = np.zeros(len(tab), dtype=DAT_DTYPE)
    for k, c in enumerate(DAT_COLS):
        rows[c] = tab[:, k]
    return rows


def build_dat(rng, dates):
    """tick.dat rows for consecutive sessions, offset to global row numbers as to_dat does, as an int64 table."""
    blocks, offset = [], 0
    for d in dates:
        tick_res, bars, n = process_session(make_session(rng, d), d)
        blocks.append(offset_session(tick_res, bars, offset))
        offset += n
    return table(np.concatenate(blocks))


def tick_scan(data, i, end, hi, lo):
    p = data[i:end, PRICE_COL]
    hit = np.flatnonzero((p >= hi) | (p <= lo))
    if len(hit) == 0:
        return 0
    return 1 if p[hit[0]] >= hi else -1


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    return build_dat(rng, [datetime.date(2024, 3, 11), datetime.date(2024, 3, 12)])


@pytest.fixture(scope='module')
def stops(data):
    return StopSearch(pack(data))


def test_child_bars_tile_their_parent(data):
    # The search relies on this: a parent's children start on its first row and end exactly at its end
    for parent, child in CHILD.items():
        if child is None:
            continue
        for i in np.flatnonzero(data[:, NEXT[parent]]):
            end, j = data[i, NEXT[parent]], i
            while j < end:
                assert data[j, NEXT[child]] > 0, f"{child} bar missing inside {parent} bar at row {i}"
                j = data[j, NEXT[child]]
            assert j == end, f"{child} bars overrun the {parent} bar at row {i}"


@pytest.mark.parametrize('freq', list(CHILD))
def test_matches_tick_scan(stops, data, freq):
    rng = np.random.default_rng(1)
    starts = np.flatnonzero(data[:, NEXT[freq]])
    sample = rng.choice(starts, size=min(300, len(starts)), replace=False)
    sides = []
    for i in sample:
        p0 = data[i, PRICE_COL]
        for _ in range(max(5, 1500 // len(sample))):
            hi = p0 + int(rng.integers(0, 80))
            lo = p0 - int(rng.integers(0, 80))
            want = tick_scan(data, i, data[i, NEXT[freq]], hi, lo)
            assert stops.first_hit(freq, i, hi, lo) == want, f"{freq} bar at row {i}, levels {hi}/{lo}"
            sides.append(want)
    assert {1, -1} <= set(sides)


def test_both_levels_in_one_second_walks_ticks(stops, data):
    starts = np.flatnonzero(data[:, NEXT['1s']])
    both = [i for i in starts if data[i, HIGH['1s']] > data[i, LOW['1s']]]
    assert len(both) > 50
    for i in both:
        hi, lo = data[i, HIGH['1s']], data[i, LOW['1s']]
        assert stops.first_hit('1s', i, hi, lo) == tick_scan(data, i, data[i, NEXT['1s']], hi, lo)


def test_clean_bar_is_skipped_even_if_a_later_bar_hits(stops, data):
    for i in np.flatnonzero(data[:, NEXT['5']]):
        hi, lo = data[i, HIGH['5']] + 1, data[i, LOW['5']] - 1
        if tick_scan(data, i, len(data), hi, lo) != 0:
            break
    else:
        pytest.fail("no 5-min bar with a later hit")
    assert stops.first_hit('5', i, hi, lo) == 0


def test_start_must_be_first_tick_of_the_bar(stops, data):
    i = int(np.flatnonzero(data[:, NEXT['60']] == 0)[0])
    p0 = data[i, PRICE_COL]
    with pytest.raises(ValueError):
        stops.first_hit('60', i, p0 + 4000, p0 - 4000)


def test_inconsistent_summaries_raise(data):
    # A 1-min bar whose summary claims both levels while none of its 15s bars reach either
    bad = data.copy()
    i = int(np.flatnonzero(bad[:, NEXT['1']])[10])
    p0 = bad[i, PRICE_COL]
    bad[i, HIGH['1']], bad[i, LOW['1']] = p0 + 4000, p0 - 4000
    with pytest.raises(ValueError):
        StopSearch(pack(bad)).first_hit('1', i, p0 + 2000, p0 - 2000)


@pytest.mark.parametrize('freq', list(CHILD))
def test_arrays_match_single_calls(stops, data, freq):
    rng = np.random.default_rng(2)
    starts = rng.choice(np.flatnonzero(data[:, NEXT[freq]]), size=2000)
    p0 = data[starts, PRICE_COL]
    hi = p0 + rng.integers(0, 80, len(starts))
    lo = p0 - rng.integers(0, 80, len(starts))
    sides = stops.first_hit_many(freq, starts, hi, lo)
    assert sides.dtype == np.int8
    assert sides.tolist() == [stops.first_hit(freq, int(s), int(h), int(l)) for s, h, l in zip(starts, hi, lo)]


def test_one_entry_gives_int_and_scalars_broadcast(stops, data):
    starts = np.flatnonzero(data[:, NEXT['15']])[:300]
    p0 = int(data[starts[0], PRICE_COL])
    side = stops.first_hit('15', starts[0], np.int64(p0 + 100), p0 - 100)
    assert type(side) is int
    # the same two levels for every entry
    sides = stops.first_hit_many('15', starts, p0 + 100, p0 - 100)
    assert sides.tolist() == [stops.first_hit('15', int(s), p0 + 100, p0 - 100) for s in starts]
    assert stops.first_hit_many('15', starts[:0], p0 + 100, p0 - 100).shape == (0,)
    with pytest.raises(ValueError):
        stops.first_hit_many('15', starts[:3], np.array([p0, p0]), p0 - 100)  # lengths 2 and 3


def test_arrays_reject_bad_starts_and_broken_summaries(stops, data):
    starts = np.flatnonzero(data[:, NEXT['1']])[:500]
    p0 = data[starts, PRICE_COL]
    with pytest.raises(ValueError):
        stops.first_hit_many('1', starts + 1, p0 + 100, p0 - 100)  # rows after a bar start
    bad = data.copy()
    i = starts[10]
    bad[i, HIGH['1']], bad[i, LOW['1']] = p0[10] + 4000, p0[10] - 4000
    with pytest.raises(ValueError):
        StopSearch(pack(bad)).first_hit_many('1', starts, p0 + 2000, p0 - 2000)


def test_load_reads_to_dat_layout(data, tmp_path):
    path = tmp_path / 'tick.dat'
    pack(data).tofile(path)
    mm = StopSearch.load(path)
    assert mm.data.shape == (len(data),)
    i = int(np.flatnonzero(data[:, NEXT['60']])[3])
    p0 = mm.price(i)
    assert mm.first_hit('60', i, p0 + 100, p0 - 100) == tick_scan(data, i, data[i, NEXT['60']], p0 + 100, p0 - 100)

    with open(path, 'ab') as f:
        f.write(b'\0' * 8)
    with pytest.raises(ValueError):
        StopSearch.load(path)


def test_zero_d_arrays_count_as_one_entry(stops, data):
    i = int(np.flatnonzero(data[:, NEXT['5']])[7])
    p0 = data[i, PRICE_COL]
    side = stops.first_hit('5', np.array(i), np.array(p0 + 100), np.array(p0 - 100))
    assert type(side) is int and side == stops.first_hit('5', i, int(p0) + 100, int(p0) - 100)


def test_price_and_bar_end_read_the_rows_as_int64(data):
    ticks = StopSearch(pack(data))
    for freq in CHILD:
        starts = np.flatnonzero(data[:, NEXT[freq]])
        assert np.array_equal(ticks.bar_end(freq, starts), data[starts, NEXT[freq]])
        assert np.array_equal(ticks.price(starts), data[starts, PRICE_COL])
    assert ticks.price(starts).dtype == np.int64 and ticks.bar_end(freq, starts).dtype == np.int64


def test_stopsearch_rejects_child_tables_that_do_not_tile(data):
    with pytest.raises(ValueError):
        StopSearch(pack(data), child={**CHILD, '15': '10'})
    with pytest.raises(ValueError):
        StopSearch(pack(data), child={**CHILD, '5': '2'})


def test_float_levels_are_exact(stops, data):
    rng = np.random.default_rng(3)
    starts = rng.choice(np.flatnonzero(data[:, NEXT['1']]), size=1500)
    p0 = data[starts, PRICE_COL].astype(float)
    upper = p0 + rng.integers(0, 40, len(starts)) + rng.uniform(-0.96, 0.96, len(starts))
    lower = p0 - rng.integers(0, 40, len(starts)) + rng.uniform(-0.96, 0.96, len(starts))
    want = [tick_scan(data, s, data[s, NEXT['1']], u, l) for s, u, l in zip(starts, upper, lower)]
    assert stops.first_hit_many('1', starts, upper, lower).tolist() == want
    assert stops.first_hit('1', int(starts[0]), float(upper[0]), float(lower[0])) == want[0]


def test_inf_means_no_level_and_nan_raises(stops, data):
    starts = np.flatnonzero(data[:, NEXT['60']])[:200]
    p0 = data[starts, PRICE_COL]
    only_lower = stops.first_hit_many('60', starts, np.inf, p0 - 100)
    assert set(only_lower.tolist()) <= {0, -1}
    assert only_lower.tolist() == [tick_scan(data, s, data[s, NEXT['60']], np.inf, l) for s, l in zip(starts, p0 - 100)]
    with pytest.raises(ValueError):
        stops.first_hit_many('60', starts, np.where(np.arange(len(starts)) == 5, np.nan, p0 + 100.0), p0 - 100)


def test_cut_or_negative_rows_raise_instead_of_reading_outside(stops, data):
    day0, day1 = np.flatnonzero(data[:, NEXT['day']])[:2]
    cut = data[:day1 - 100]  # ends inside the first session: its day bar points past the end
    p0 = data[day0, PRICE_COL]
    with pytest.raises(ValueError):
        StopSearch(pack(cut)).first_hit('day', int(day0), int(p0) + 10**7, int(p0) - 10**7)
    with pytest.raises(ValueError):
        StopSearch(pack(cut)).first_hit_many('day', np.array([day0]), p0 + 10**7, p0 - 10**7)
    with pytest.raises(ValueError):
        stops.first_hit('1', -1, int(p0) + 100, int(p0) - 100)
    with pytest.raises(ValueError):
        stops.first_hit_many('1', np.array([-1, 0]), p0 + 100, p0 - 100)


def test_child_table_cycles_raise(data):
    for bad in ({**CHILD, '1s': '1s'}, {**CHILD, '5': '5'}):
        with pytest.raises(ValueError):
            StopSearch(pack(data), child=bad)


def test_params_import_does_not_load_numba():
    code = "import sys; from stop_search.params import FREQS, DAT_COLS; assert 'numba' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True, cwd=os.path.join(os.path.dirname(__file__), '..'))


def test_each_call_rejects_the_other_kind_of_input(stops, data):
    starts = np.flatnonzero(data[:, NEXT['1']])[:5]
    p0 = int(data[starts[0], PRICE_COL])
    with pytest.raises(TypeError):
        stops.first_hit('1', starts, p0 + 40, p0 - 20)             # many entries -> first_hit_many
    with pytest.raises(TypeError):
        stops.first_hit('1', int(starts[0]), np.array([p0 + 40]), p0 - 20)
    with pytest.raises(TypeError):
        stops.first_hit_many('1', int(starts[0]), p0 + 40, p0 - 20)  # one entry -> first_hit
    sides = stops.first_hit_many('1', starts.tolist(), p0 + 40, p0 - 20)  # plain lists are fine
    assert sides.dtype == np.int8 and sides.tolist() == [stops.first_hit('1', int(s), p0 + 40, p0 - 20) for s in starts]


def test_unknown_bar_size_raises_a_clear_error(stops, data):
    i = int(np.flatnonzero(data[:, NEXT['15']])[0])
    p0 = int(data[i, PRICE_COL])
    for bad in (15, '2', None):
        for call in (lambda: stops.first_hit(bad, i, p0 + 40, p0 - 20),
                     lambda: stops.first_hit_many(bad, [i], p0 + 40, p0 - 20),
                     lambda: stops.bar_end(bad, i)):
            with pytest.raises(ValueError, match="unknown bar size"):
                call()


def test_load_defaults_to_params_dat_path(data, tmp_path, monkeypatch):
    from stop_search import DAT_PATH
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.dirname(DAT_PATH))
    pack(data).tofile(DAT_PATH)
    stops = StopSearch.load()
    assert stops.data.filename == os.path.abspath(DAT_PATH) and stops.data.shape == (len(data),)


def test_paths_come_from_params_and_are_relative():
    # Every path lives in params/params.py; StopSearch.load() opens the tick.dat to_dat.py writes
    import params
    from stop_search import DAT_PATH
    from data_pipeline.to_dat import DEFAULT_OUT as out
    assert DAT_PATH == params.TICK_DATA_PATH
    assert os.path.normpath(os.path.join(out, 'tick.dat')) == os.path.normpath(DAT_PATH)
    # Relative to the repo root: tests like the one above chdir into a temp folder and write fake files at these
    # paths, so an absolute path would overwrite the real data
    paths = {k: v for k, v in vars(params).items() if k.endswith(('_PATH', '_DIR')) and isinstance(v, str)}
    assert paths and all(not os.path.isabs(p) and not p.startswith('~') for p in paths.values()), paths
