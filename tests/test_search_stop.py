import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import numpy as np
import pandas as pd
import pytest

from to_dat import FREQS, DAT_COLS, process_session
from search_stop import search_stop, load_dat, CHILD, PRICE_COL

NEXT = {f: DAT_COLS.index(f'next_ind_{f}') for f in FREQS}
HIGH = {f: DAT_COLS.index(f'high_{f}') for f in FREQS}
LOW = {f: DAT_COLS.index(f'low_{f}') for f in FREQS}


def make_session(rng, date):
    """One session of step-1 ticks: bursts and quiet gaps, several ticks per second, 0.25-pt steps."""
    secs = np.cumsum(rng.exponential(rng.choice([0.1, 1.0, 20.0], size=40_000)))
    secs = secs[secs < 23 * 3600].astype(np.int64)
    n = len(secs)
    df = pd.DataFrame({
        'ts': pd.Timestamp(date) + pd.to_timedelta(secs, unit='s'),
        'price': 500_000 + 25 * np.cumsum(rng.integers(-1, 2, n)),
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
            hi = p0 + 25 * int(rng.integers(0, 80))
            lo = p0 - 25 * int(rng.integers(0, 80))
            want = tick_scan(data, i, data[i, NEXT[freq]], hi, lo)
            assert search_stop(data, hi, lo, freq, i) == want, f"{freq} bar at row {i}, levels {hi}/{lo}"
            sides.append(want)
    assert {1, -1} <= set(sides)


def test_both_levels_in_one_second_walks_ticks(data):
    starts = np.flatnonzero(data[:, NEXT['1s']])
    both = [i for i in starts if data[i, HIGH['1s']] > data[i, LOW['1s']]]
    assert len(both) > 50
    for i in both:
        hi, lo = data[i, HIGH['1s']], data[i, LOW['1s']]
        assert search_stop(data, hi, lo, '1s', i) == tick_scan(data, i, data[i, NEXT['1s']], hi, lo)


def test_clean_bar_is_skipped_even_if_a_later_bar_hits(data):
    for i in np.flatnonzero(data[:, NEXT['5']]):
        hi, lo = data[i, HIGH['5']] + 25, data[i, LOW['5']] - 25
        if tick_scan(data, i, len(data), hi, lo) != 0:
            break
    else:
        pytest.fail("no 5-min bar with a later hit")
    assert search_stop(data, hi, lo, '5', i) == 0


def test_start_must_be_first_tick_of_the_bar(data):
    i = int(np.flatnonzero(data[:, NEXT['60']] == 0)[0])
    p0 = data[i, PRICE_COL]
    with pytest.raises(ValueError):
        search_stop(data, p0 + 100_000, p0 - 100_000, '60', i)


def test_inconsistent_summaries_raise(data):
    # A 1-min bar whose summary claims both levels while none of its 15s bars reach either
    bad = data.copy()
    i = int(np.flatnonzero(bad[:, NEXT['1']])[10])
    p0 = bad[i, PRICE_COL]
    bad[i, HIGH['1']], bad[i, LOW['1']] = p0 + 100_000, p0 - 100_000
    with pytest.raises(ValueError):
        search_stop(bad, p0 + 50_000, p0 - 50_000, '1', i)


@pytest.mark.parametrize('freq', list(CHILD))
def test_arrays_match_single_calls(data, freq):
    rng = np.random.default_rng(2)
    starts = rng.choice(np.flatnonzero(data[:, NEXT[freq]]), size=2000)
    p0 = data[starts, PRICE_COL]
    hi = p0 + 25 * rng.integers(0, 80, len(starts))
    lo = p0 - 25 * rng.integers(0, 80, len(starts))
    sides = search_stop(data, hi, lo, freq, starts)
    assert sides.dtype == np.int8
    assert sides.tolist() == [search_stop(data, int(h), int(l), freq, int(s)) for s, h, l in zip(starts, hi, lo)]


def test_one_entry_gives_int_and_scalars_broadcast(data):
    starts = np.flatnonzero(data[:, NEXT['15']])[:300]
    p0 = int(data[starts[0], PRICE_COL])
    side = search_stop(data, np.int64(p0 + 2500), p0 - 2500, '15', starts[0])
    assert type(side) is int
    # the same two levels for every entry
    sides = search_stop(data, p0 + 2500, p0 - 2500, '15', starts)
    assert sides.tolist() == [search_stop(data, p0 + 2500, p0 - 2500, '15', int(s)) for s in starts]
    assert search_stop(data, p0 + 2500, p0 - 2500, '15', starts[:0]).shape == (0,)
    with pytest.raises(ValueError):
        search_stop(data, np.array([p0, p0]), p0 - 2500, '15', starts[:3])  # lengths 2 and 3


def test_arrays_reject_bad_starts_and_broken_summaries(data):
    starts = np.flatnonzero(data[:, NEXT['1']])[:500]
    p0 = data[starts, PRICE_COL]
    with pytest.raises(ValueError):
        search_stop(data, p0 + 2500, p0 - 2500, '1', starts + 1)  # rows after a bar start
    bad = data.copy()
    i = starts[10]
    bad[i, HIGH['1']], bad[i, LOW['1']] = p0[10] + 100_000, p0[10] - 100_000
    with pytest.raises(ValueError):
        search_stop(bad, p0 + 50_000, p0 - 50_000, '1', starts)


def test_load_dat_reads_to_dat_layout(data, tmp_path):
    path = tmp_path / 'tick.dat'
    data.tofile(path)
    mm = load_dat(path)
    assert mm.shape == data.shape
    i = int(np.flatnonzero(mm[:, NEXT['60']])[3])
    p0 = mm[i, PRICE_COL]
    assert search_stop(mm, p0 + 2500, p0 - 2500, '60', i) == tick_scan(data, i, data[i, NEXT['60']], p0 + 2500, p0 - 2500)

    with open(path, 'ab') as f:
        f.write(b'\0' * 8)
    with pytest.raises(ValueError):
        load_dat(path)


def test_zero_d_arrays_count_as_one_entry(data):
    i = int(np.flatnonzero(data[:, NEXT['5']])[7])
    p0 = data[i, PRICE_COL]
    side = search_stop(data, np.array(p0 + 2500), np.array(p0 - 2500), '5', np.array(i))
    assert type(side) is int and side == search_stop(data, int(p0) + 2500, int(p0) - 2500, '5', i)
