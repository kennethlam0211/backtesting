import os
import sys
import datetime
import multiprocessing
import subprocess
import pickle
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest

from data_pipeline.to_dat import process_session
from stop_search import first_hit, first_hit_many, load_dat, StopSearch, CHILD, DAT_COLS, FREQS, PRICE_COL

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


def build_dat(rng, dates):
    """tick.dat rows for consecutive sessions, offset to global indices the way to_dat.write_result does."""
    blocks, offset = [], 0
    for d in dates:
        tick_res, _, n = process_session(make_session(rng, d), d)
        start = tick_res['start_ind'].values
        tick_res['start_ind'] = np.where(start != -1, start + offset, 0)
        for f in FREQS:
            col = f'next_ind_{f}'
            tick_res[col] = np.where(tick_res[col] > 0, tick_res[col] + offset, 0)
        blocks.append(tick_res[DAT_COLS].values.astype(np.int64))
        offset += n
    return np.concatenate(blocks)


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
def test_matches_tick_scan(data, freq):
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
            assert first_hit(data, freq, i, hi, lo) == want, f"{freq} bar at row {i}, levels {hi}/{lo}"
            sides.append(want)
    assert {1, -1} <= set(sides)


def test_both_levels_in_one_second_walks_ticks(data):
    starts = np.flatnonzero(data[:, NEXT['1s']])
    both = [i for i in starts if data[i, HIGH['1s']] > data[i, LOW['1s']]]
    assert len(both) > 50
    for i in both:
        hi, lo = data[i, HIGH['1s']], data[i, LOW['1s']]
        assert first_hit(data, '1s', i, hi, lo) == tick_scan(data, i, data[i, NEXT['1s']], hi, lo)


def test_clean_bar_is_skipped_even_if_a_later_bar_hits(data):
    for i in np.flatnonzero(data[:, NEXT['5']]):
        hi, lo = data[i, HIGH['5']] + 1, data[i, LOW['5']] - 1
        if tick_scan(data, i, len(data), hi, lo) != 0:
            break
    else:
        pytest.fail("no 5-min bar with a later hit")
    assert first_hit(data, '5', i, hi, lo) == 0


def test_start_must_be_first_tick_of_the_bar(data):
    i = int(np.flatnonzero(data[:, NEXT['60']] == 0)[0])
    p0 = data[i, PRICE_COL]
    with pytest.raises(ValueError):
        first_hit(data, '60', i, p0 + 4000, p0 - 4000)


def test_inconsistent_summaries_raise(data):
    # A 1-min bar whose summary claims both levels while none of its 15s bars reach either
    bad = data.copy()
    i = int(np.flatnonzero(bad[:, NEXT['1']])[10])
    p0 = bad[i, PRICE_COL]
    bad[i, HIGH['1']], bad[i, LOW['1']] = p0 + 4000, p0 - 4000
    with pytest.raises(ValueError):
        first_hit(bad, '1', i, p0 + 2000, p0 - 2000)


@pytest.mark.parametrize('freq', list(CHILD))
def test_arrays_match_single_calls(data, freq):
    rng = np.random.default_rng(2)
    starts = rng.choice(np.flatnonzero(data[:, NEXT[freq]]), size=2000)
    p0 = data[starts, PRICE_COL]
    hi = p0 + rng.integers(0, 80, len(starts))
    lo = p0 - rng.integers(0, 80, len(starts))
    sides = first_hit_many(data, freq, starts, hi, lo)
    assert sides.dtype == np.int8
    assert sides.tolist() == [first_hit(data, freq, int(s), int(h), int(l)) for s, h, l in zip(starts, hi, lo)]


def test_one_entry_gives_int_and_scalars_broadcast(data):
    starts = np.flatnonzero(data[:, NEXT['15']])[:300]
    p0 = int(data[starts[0], PRICE_COL])
    side = first_hit(data, '15', starts[0], np.int64(p0 + 100), p0 - 100)
    assert type(side) is int
    # the same two levels for every entry
    sides = first_hit_many(data, '15', starts, p0 + 100, p0 - 100)
    assert sides.tolist() == [first_hit(data, '15', int(s), p0 + 100, p0 - 100) for s in starts]
    assert first_hit_many(data, '15', starts[:0], p0 + 100, p0 - 100).shape == (0,)
    with pytest.raises(ValueError):
        first_hit_many(data, '15', starts[:3], np.array([p0, p0]), p0 - 100)  # lengths 2 and 3


def test_arrays_reject_bad_starts_and_broken_summaries(data):
    starts = np.flatnonzero(data[:, NEXT['1']])[:500]
    p0 = data[starts, PRICE_COL]
    with pytest.raises(ValueError):
        first_hit_many(data, '1', starts + 1, p0 + 100, p0 - 100)  # rows after a bar start
    bad = data.copy()
    i = starts[10]
    bad[i, HIGH['1']], bad[i, LOW['1']] = p0[10] + 4000, p0[10] - 4000
    with pytest.raises(ValueError):
        first_hit_many(bad, '1', starts, p0 + 2000, p0 - 2000)


def test_load_dat_reads_to_dat_layout(data, tmp_path):
    path = tmp_path / 'tick.dat'
    data.tofile(path)
    mm = load_dat(path)
    assert mm.shape == data.shape
    i = int(np.flatnonzero(mm[:, NEXT['60']])[3])
    p0 = mm[i, PRICE_COL]
    assert first_hit(mm, '60', i, p0 + 100, p0 - 100) == tick_scan(data, i, data[i, NEXT['60']], p0 + 100, p0 - 100)

    with open(path, 'ab') as f:
        f.write(b'\0' * 8)
    with pytest.raises(ValueError):
        load_dat(path)


def test_zero_d_arrays_count_as_one_entry(data):
    i = int(np.flatnonzero(data[:, NEXT['5']])[7])
    p0 = data[i, PRICE_COL]
    side = first_hit(data, '5', np.array(i), np.array(p0 + 100), np.array(p0 - 100))
    assert type(side) is int and side == first_hit(data, '5', i, int(p0) + 100, int(p0) - 100)


class Backtester:
    """Stand-in for the class that owns a StopSearch."""

    def __init__(self, ticks):
        self.ticks = ticks

    def label(self, freq, tp, sl):
        starts = self.ticks.bar_starts(freq)
        entry = self.ticks.price(starts)
        return self.ticks.first_hit_many(freq, starts, entry + tp, entry - sl)


def _label_in_worker(bt, freq):
    return bt.label(freq, 40, 20)


@pytest.fixture(scope='module')
def dat_file(data, tmp_path_factory):
    path = tmp_path_factory.mktemp('dat') / 'tick.dat'
    data.tofile(path)
    return path


def test_stopsearch_matches_function(data):
    ticks = StopSearch(data)
    assert len(ticks) == len(data)
    for freq in CHILD:
        starts = ticks.bar_starts(freq)
        assert np.array_equal(starts, np.flatnonzero(data[:, NEXT[freq]]))
        assert ticks.bar_starts(freq) is starts  # cached
        assert np.array_equal(ticks.bar_end(freq, starts), data[starts, NEXT[freq]])
        entry = ticks.price(starts)
        want = first_hit_many(data, freq, starts, entry + 40, entry - 20)
        assert np.array_equal(ticks.first_hit_many(freq, starts, entry + 40, entry - 20), want)
        assert ticks.first_hit(freq, int(starts[0]), int(entry[0]) + 40, int(entry[0]) - 20) == want[0]


def test_stopsearch_rejects_child_tables_that_do_not_tile(data):
    with pytest.raises(ValueError):
        StopSearch(data, child={**CHILD, '15': '10'})
    with pytest.raises(ValueError):
        StopSearch(data, child={**CHILD, '5': '2'})


def test_stopsearch_pickles_by_path(data, dat_file):
    ticks = StopSearch.load(dat_file)
    ticks.bar_starts('1')
    blob = pickle.dumps(Backtester(ticks))
    assert len(blob) < 10_000 < data.nbytes  # the path, not the ticks
    bt = pickle.loads(blob)
    assert isinstance(bt.ticks.data, np.memmap)
    assert np.array_equal(bt.label('1', 40, 20), Backtester(StopSearch(data)).label('1', 40, 20))


def test_in_memory_stopsearch_pickles_with_its_ticks(data):
    ticks = pickle.loads(pickle.dumps(StopSearch(data)))
    assert np.array_equal(ticks.data, data)


def test_owner_class_works_in_worker_processes(dat_file):
    bt = Backtester(StopSearch.load(dat_file))
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(_label_in_worker, [bt, bt], ['1', '15']))
    assert np.array_equal(results[0], bt.label('1', 40, 20))
    assert np.array_equal(results[1], bt.label('15', 40, 20))


def test_float_levels_are_exact(data):
    rng = np.random.default_rng(3)
    starts = rng.choice(np.flatnonzero(data[:, NEXT['1']]), size=1500)
    p0 = data[starts, PRICE_COL].astype(float)
    upper = p0 + rng.integers(0, 40, len(starts)) + rng.uniform(-0.96, 0.96, len(starts))
    lower = p0 - rng.integers(0, 40, len(starts)) + rng.uniform(-0.96, 0.96, len(starts))
    want = [tick_scan(data, s, data[s, NEXT['1']], u, l) for s, u, l in zip(starts, upper, lower)]
    assert first_hit_many(data, '1', starts, upper, lower).tolist() == want
    assert first_hit(data, '1', int(starts[0]), float(upper[0]), float(lower[0])) == want[0]


def test_inf_means_no_level_and_nan_raises(data):
    starts = np.flatnonzero(data[:, NEXT['60']])[:200]
    p0 = data[starts, PRICE_COL]
    only_lower = first_hit_many(data, '60', starts, np.inf, p0 - 100)
    assert set(only_lower.tolist()) <= {0, -1}
    assert only_lower.tolist() == [tick_scan(data, s, data[s, NEXT['60']], np.inf, l) for s, l in zip(starts, p0 - 100)]
    with pytest.raises(ValueError):
        first_hit_many(data, '60', starts, np.where(np.arange(len(starts)) == 5, np.nan, p0 + 100.0), p0 - 100)


def test_cut_or_negative_rows_raise_instead_of_reading_outside(data):
    day0, day1 = np.flatnonzero(data[:, NEXT['day']])[:2]
    cut = data[:day1 - 100]  # ends inside the first session: its day bar points past the end
    p0 = data[day0, PRICE_COL]
    with pytest.raises(ValueError):
        first_hit(cut, 'day', int(day0), int(p0) + 10**7, int(p0) - 10**7)
    with pytest.raises(ValueError):
        first_hit_many(cut, 'day', np.array([day0]), p0 + 10**7, p0 - 10**7)
    with pytest.raises(ValueError):
        first_hit(data, '1', -1, int(p0) + 100, int(p0) - 100)
    with pytest.raises(ValueError):
        first_hit_many(data, '1', np.array([-1, 0]), p0 + 100, p0 - 100)


def test_bar_starts_cache_is_read_only(data):
    starts = StopSearch(data).bar_starts('1')
    with pytest.raises(ValueError):
        starts += 1


def test_child_table_cycles_raise(data):
    for bad in ({**CHILD, '1s': '1s'}, {**CHILD, '5': '5'}):
        with pytest.raises(ValueError):
            StopSearch(data, child=bad)


def test_params_import_does_not_load_numba():
    code = "import sys; from stop_search.params import FREQS, DAT_COLS; assert 'numba' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True, cwd=os.path.join(os.path.dirname(__file__), '..'))


def test_pickle_uses_absolute_path_and_memmap_file(data, dat_file, tmp_path, monkeypatch):
    monkeypatch.chdir(dat_file.parent)
    ticks = StopSearch.load(dat_file.name)  # relative path
    assert os.path.isabs(ticks.path)
    monkeypatch.chdir(tmp_path)  # a worker with another working directory
    assert np.array_equal(pickle.loads(pickle.dumps(ticks)).data, data)
    # built from load_dat: the memmap knows its file, so pickling still carries only the path
    assert len(pickle.dumps(StopSearch(load_dat(dat_file)))) < 10_000
    # a slice of the memmap is not the whole file: pickled with its ticks
    part = load_dat(dat_file)[:100]
    assert np.array_equal(pickle.loads(pickle.dumps(StopSearch(part))).data, part)


def test_unpickle_refuses_a_changed_file(data, tmp_path):
    path = tmp_path / 'tick.dat'
    data.tofile(path)
    blob = pickle.dumps(StopSearch.load(path))
    data[:10].tofile(path)  # to_dat.py wrote a new tick.dat
    with pytest.raises(ValueError):
        pickle.loads(blob)


def test_each_function_rejects_the_other_kind_of_input(data):
    starts = np.flatnonzero(data[:, NEXT['1']])[:5]
    p0 = int(data[starts[0], PRICE_COL])
    with pytest.raises(TypeError):
        first_hit(data, '1', starts, p0 + 40, p0 - 20)             # many entries -> first_hit_many
    with pytest.raises(TypeError):
        first_hit(data, '1', int(starts[0]), np.array([p0 + 40]), p0 - 20)
    with pytest.raises(TypeError):
        first_hit_many(data, '1', int(starts[0]), p0 + 40, p0 - 20)  # one entry -> first_hit
    sides = first_hit_many(data, '1', starts.tolist(), p0 + 40, p0 - 20)  # plain lists are fine
    assert sides.dtype == np.int8 and sides.tolist() == [first_hit(data, '1', int(s), p0 + 40, p0 - 20) for s in starts]


def test_unknown_bar_size_raises_a_clear_error(data):
    stops = StopSearch(data)
    i = int(np.flatnonzero(data[:, NEXT['15']])[0])
    p0 = int(data[i, PRICE_COL])
    for bad in (15, '2', None):
        for call in (lambda: first_hit(data, bad, i, p0 + 40, p0 - 20),
                     lambda: first_hit_many(data, bad, [i], p0 + 40, p0 - 20),
                     lambda: stops.first_hit(bad, i, p0 + 40, p0 - 20),
                     lambda: stops.bar_starts(bad)):
            with pytest.raises(ValueError, match="unknown bar size"):
                call()


def test_load_defaults_to_params_dat_path(data, tmp_path, monkeypatch):
    from stop_search import DAT_PATH
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.dirname(DAT_PATH))
    data.tofile(DAT_PATH)
    stops = StopSearch.load()
    assert stops.path == os.path.abspath(DAT_PATH) and stops.data.shape == data.shape


def _default(script, option):
    """An argparse default from a data_pipeline script, read from its source."""
    src = open(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline', script)).read()
    return re.search(rf'"{option}".*?default="([^"]+)"', src).group(1)


def test_default_paths_agree_with_to_dat():
    # to_dat.py keeps its own output path; StopSearch.load() and step 3 must follow it
    from stop_search import DAT_PATH
    from data_pipeline.to_dat import DEFAULT_OUT as out
    assert os.path.normpath(os.path.join(out, 'tick.dat')) == os.path.normpath(DAT_PATH), "update DAT_PATH in stop_search/params.py"
    assert os.path.normpath(_default('data_preprocessing(template).py', '--data-dir')) == os.path.normpath(out), "update --data-dir in data_preprocessing(template).py"
