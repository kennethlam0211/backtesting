"""
Sweeps: every signal of a preset, long and short as separate strategies, over a stop grid, one table of trades per
trade freq. The training data is loaded once per freq; each signal is one ensemble of one (config.ENSEMBLE) through
run_grid, with the news and volatility-band rules of config.py.

    python -m backtesting.sweep rsi --freq 5,10,15,30,60          # RSI levels, normal and reversed, fixed stops
    python -m backtesting.sweep ud_near --freq 5,10,15,30,60      # near any of a freq's last 5 U/D pivots
    python -m backtesting.sweep ud_range                          # 1-min execution, near a freq's last-two-pivot range
    python -m backtesting.sweep ud_range --dists 20 --out results/sweep/ud_range_5pt   # within 5 points instead
    python -m backtesting.sweep ud_range --only '_30_12_2_short$' --flat-news --out results/sweep/ud_range_flat_news
    python -m backtesting.sweep ud_range --pivot-freqs session --out results/sweep/ud_range_session   # session pivots
    python -m backtesting.sweep ud_level                          # 1-min, at 30 / 50 / 60% inside the range
    python -m backtesting.sweep ud_near_level                     # 1-min, at 30 / 50 / 60% inside the last 4 swings
    python -m backtesting.sweep ud_near --freq 1 --out results/sweep/ud_near_1min   # version 2 on 1-min execution
    python -m backtesting.sweep ud_ends                           # 1-min, long / short at the high end, at the low end

Presets:
- rsi: rsi_long / rsi_short at 1/99, 2/98, 3/97, 5/95, 10/90, 15/85, normal and reversed, on the trade freq's RSI.
- ud_near: strategy.ud_near with pivots last1..last5 of every freq from the trade freq up, within 2 / 4 / 8 units.
- ud_range: strategy.ud_near with pivots last1..last2 (the range) of 5 / 10 / 15 / 30 / 60-min and day, within
  2 / 4 / 8 units; each trade also keeps `near`, every one of those freqs whose range end was that close at its signal
  bar (e.g. '5,15'), and `range_freq`.
- ud_ends: strategy.ud_end, the range's ends apart: long and short at the high end (the higher of last1 / last2), long
  and short at the low end, of the same freqs and distances (1-min by default).
- ud_level: strategy.ud_level inside the range: 30 / 50 / 60% of the way from last1 back to last2 (1-min by default).
- ud_near_level: the same inside any of the last 4 swings (between consecutive pivots of last1..last5).

Writes results/sweep/{preset}/trades_{freq}.parquet (signal, freq, stops, side, entry / exit, result, pnl, gross_usd,
net_usd, hour, and the preset's labels) and summary_{freq}.parquet (per signal x stops: trades, net, net per trade, win
rate, net per trade before / from 2026). Then: python -m backtesting.pairs {preset}.
"""
import argparse
import os
import re
import time

import polars as pl

from backtesting import config
from backtesting.backtest import read_training_data, run_grid, vol_ok
from backtesting.stops import grid, vol_grid
from backtesting.strategy import (
    Strategy,
    ensemble,
    rsi_long,
    rsi_short,
    ud_end,
    ud_level,
    ud_near,
)
from params import RESULTS_DIR, TICK_DATA_PATH
from stop_search import StopSearch

MINUTES = {'1': 1, '5': 5, '10': 10, '15': 15, '30': 30, '60': 60, 'session': 480, 'day': 1440}
RSI_LEVELS = [(1, 99), (2, 98), (3, 97), (5, 95), (10, 90), (15, 85)]
UD_FREQS = ['5', '10', '15', '30', '60', 'day']
UD_DISTS = [2, 4, 8]  # units: 0.5, 1, 2 points
UD_LEVELS = [0.3, 0.5, 0.6]  # ud_level presets: how far from a swing's newer pivot back to its older one
FIXED = grid(tp=[16, 24, 32, 40, 60, 100], sl=[16, 24, 32, 40, 60, 100])
VOL_MULTIPLES = [1, 2, 3, 4, 6]
TEST_FROM = pl.datetime(2026, 1, 1)
COLUMNS = ['signal', 'freq', 'stops', 'side', 'entry_ts', 'entry_px', 'exit_ts', 'result', 'pnl', 'gross_usd', 'net_usd',
           'hour']


def signals(preset: str, freq: str, dists=UD_DISTS, pivot_freqs=UD_FREQS) -> list[Strategy]:
    """The preset's strategies on `freq` bars, one per side (`dists`: the UD presets' distances, units; `pivot_freqs`:
    the timeframes whose pivots they use)."""
    if preset == 'rsi':
        return [f(freq, lvl, rev) for lo, hi in RSI_LEVELS for rev in (False, True)
                for f, lvl in ((rsi_long, lo), (rsi_short, hi))]
    if preset == 'ud_near':
        return [ud_near(freq, pf, d, (1, 5), sd) for pf in pivot_freqs if MINUTES[pf] >= MINUTES[freq]
                for d in dists for sd in (1, -1)]
    if preset == 'ud_range':
        return [ud_near(freq, pf, d, (1, 2), sd) for pf in pivot_freqs for d in dists for sd in (1, -1)]
    if preset == 'ud_ends':  # the range's two ends apart: buys and sells at the high end, buys and sells at the low end
        return [ud_end(freq, pf, d, e, sd) for pf in pivot_freqs for e in ('high', 'low') for d in dists for sd in (1, -1)]
    if preset in ('ud_level', 'ud_near_level'):  # inside the range (last1-last2) / inside any of the last 4 swings
        swings = (1, 1) if preset == 'ud_level' else (1, 4)
        return [ud_level(freq, pf, d, f, swings, sd) for pf in pivot_freqs if MINUTES[pf] >= MINUTES[freq]
                for f in UD_LEVELS for d in dists for sd in (1, -1)]
    raise ValueError(f"unknown preset {preset!r}: rsi, ud_near, ud_range, ud_ends, ud_level or ud_near_level")


def near_label(freq: str, dist: int) -> pl.Expr:
    """Every UD_FREQS freq whose last1 or last2 is within `dist` of `freq`'s close, e.g. '5,15' ('' when none)."""
    near = [pl.when(pl.any_horizontal([(pl.col(f'close_{freq}') - pl.col(f'20_UD_last{k}_{f}')).abs() <= dist
                                       for k in (1, 2)]).fill_null(False)).then(pl.lit(f)) for f in UD_FREQS]
    return pl.concat_str(near, separator=',', ignore_nulls=True).alias(f'near_{dist}')


def stop_function(kind: str, vol_column: str):
    """The stop grid: 'fixed' (units), 'vol' (multiples of vol_column) or 'both' (each signal with both grids)."""
    by_vol = vol_grid(tp=VOL_MULTIPLES, sl=VOL_MULTIPLES, column=vol_column)
    if kind == 'fixed':
        return FIXED
    if kind == 'vol':
        return by_vol
    return lambda sig: pl.concat([FIXED(sig), by_vol(sig)], how='diagonal_relaxed')


def sweep_freq(stops: StopSearch, preset: str, freq: str, stop_kind: str, entry: str, dists=UD_DISTS, only=None,
               flat_news=False, pivot_freqs=UD_FREQS) -> pl.DataFrame:
    """Every signal of the preset on `freq` bars (only: those whose name matches this regex): all their trades, one
    table. flat_news: also flat at every news window (an open trade exits when one starts)."""
    t0 = time.time()
    vol_column, news = f'20_std_{freq}', f'news_{freq}'
    strategies = [s for s in signals(preset, freq, dists, pivot_freqs) if only is None or re.search(only, s.name)]
    extra = [f'20_UD_last{k}_{f}' for f in UD_FREQS for k in (1, 2)] if preset == 'ud_range' else []
    # The traded freq's own labels (backtest.LABEL_COLS is fixed to config.FREQ when it is imported)
    labels = ['session', 'rth', 'hour', news]
    df = read_training_data(freq=freq, columns=list(dict.fromkeys(
        [c for s in strategies for c in s.columns] + extra + [*labels, vol_column])))
    entry_ok = None
    if config.VOL_BAND is not None:
        df, entry_ok = df.with_columns(_vol_ok=vol_ok(df, vol_column)), '_vol_ok'
    if preset == 'ud_range':
        df = df.with_columns(*[near_label(freq, d) for d in dists])
    stop = stop_function(stop_kind, vol_column)
    print(f'{preset} {freq}: {len(df):,} bars, {len(strategies)} signals', flush=True)
    parts = []
    for s in strategies:
        strategy = ensemble([s], config.ENSEMBLE, s.side, config.SEARCH_MODE)
        d = strategy.signals(df)
        near = [f"near_{s.params['dist']}"] if preset == 'ud_range' else []
        trades = run_grid(stops, d, stop, (*strategy.keep, *near), search_mode=config.SEARCH_MODE, no_entry=[news],
                          pattern='signals', freq=freq, entry_ok=entry_ok, entry=entry, flat_on=[news] if flat_news else [])
        labels = ([pl.lit(s.params['pivot_freq']).alias('range_freq'), pl.col(near[0]).alias('near')]
                  if preset == 'ud_range' else [])
        parts.append(trades.select(pl.lit(s.name).alias('signal'), pl.lit(freq).alias('freq'),
                                   *[c for c in COLUMNS if c not in ('signal', 'freq')], *labels))
    print(f'{preset} {freq}: done in {time.time() - t0:.0f}s', flush=True)
    return pl.concat(parts, how='diagonal_relaxed')


def summary(trades: pl.DataFrame) -> pl.DataFrame:
    """Per signal x stops: trades, net, net per trade, win rate, net per trade before / from TEST_FROM."""
    n, late = pl.col('net_usd'), pl.col('entry_ts') >= TEST_FROM
    return (trades.group_by('signal', 'freq', 'side', 'stops')
            .agg(pl.len().alias('trades'), n.sum().alias('net'), n.mean().alias('net_pt'), (n > 0).mean().alias('win'),
                 n.filter(~late).mean().alias('net_pt_in'), n.filter(late).mean().alias('net_pt_2026'),
                 late.sum().alias('trades_2026'))
            .sort('net', descending=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('preset', choices=('rsi', 'ud_near', 'ud_range', 'ud_ends', 'ud_level', 'ud_near_level'))
    parser.add_argument('--freq', default=None, help="trade freqs, comma-separated (default: '1' for ud_range and the "
                                                     "ud_level presets, else 5,10,15,30,60)")
    parser.add_argument('--stops', choices=('fixed', 'vol', 'both'), default=None,
                        help='stop grid (default: fixed for rsi, both for the UD presets)')
    parser.add_argument('--entry', choices=('limit', 'market'), default='limit')
    parser.add_argument('--dists', default=None, help='UD presets: distances to the levels in units, comma-separated '
                                                      '(default 2,4,8 = 0.5, 1, 2 points; 20 = 5 points)')
    parser.add_argument('--out', default=None, help='output folder (default: results/sweep/{preset})')
    parser.add_argument('--only', default=None, help='only the signals whose name matches this regex')
    parser.add_argument('--flat-news', action='store_true', help='flat at every news window: an open trade exits')
    parser.add_argument('--pivot-freqs', default=None, help="UD presets: the pivots' timeframes, comma-separated "
                                                            f"(default {','.join(UD_FREQS)}; also session)")
    args = parser.parse_args(argv)
    one_min = args.preset in ('ud_range', 'ud_ends', 'ud_level', 'ud_near_level')
    freqs = (args.freq or ('1' if one_min else '5,10,15,30,60')).split(',')
    stop_kind = args.stops or ('fixed' if args.preset == 'rsi' else 'both')
    dists = [int(d) for d in args.dists.split(',')] if args.dists else UD_DISTS
    out = args.out or os.path.join(RESULTS_DIR, 'sweep', args.preset)
    os.makedirs(out, exist_ok=True)

    stops = StopSearch.load(TICK_DATA_PATH)
    for freq in freqs:  # one at a time: every freq walks tick.dat, and parallel runs fight over the page cache
        trades = sweep_freq(stops, args.preset, freq, stop_kind, args.entry, dists, args.only, args.flat_news,
                            args.pivot_freqs.split(',') if args.pivot_freqs else UD_FREQS)
        trades.write_parquet(os.path.join(out, f'trades_{freq}.parquet'))
        summary(trades).write_parquet(os.path.join(out, f'summary_{freq}.parquet'))
    print(f'trades and summaries in {out}/; next: python -m backtesting.pairs {out}')


if __name__ == '__main__':
    main()
