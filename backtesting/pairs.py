"""
Same-condition long + short pairs of a sweep (backtesting/sweep.py's output): per condition (a signal without its side
suffix: `_long` / `_short`, or rsi's `_reversed`), every long run x every short run (each with its own stops, at least
--min-trades trades), summed by session and scored by Calmar (net $ per year / max drawdown of the daily cumulative
net). The best pair per condition is ranked; the top ones are plotted and their statistics printed.

    python -m backtesting.pairs ud_range                   # results/sweep/ud_range/
    python -m backtesting.pairs ud_range --breakdown       # plus net per trade by session, regular hours and `near`
    python -m backtesting.pairs limit --positive-legs      # only legs net positive on their own (the RSI study's rule)
    python -m backtesting.pairs limit --positive-legs --portfolio   # pairs of different conditions added for Calmar
    python -m backtesting.pairs ud_range --look ud         # the UD plots' look: every session, trades and win rate
    python -m backtesting.pairs limit --positive-legs --portfolio --holdout   # pick on 2020-2025 only, test on 2026
    python -m backtesting.pairs ud_range --holdout --singles  # no pairs: each side's best runs, picked before 2026
    python -m backtesting.pairs ud_range --holdout --singles --only '_(4|8)_(long|short)$' --tag 1-2pt   # a subset
    python -m backtesting.pairs ud_range --combo ud_near_1_day_12_8_long:16/60 ud_near_1_30_12_2_short:1x/6x
    python -m backtesting.pairs ud_range ud_near --combo ud_near_5_day_15_8_long:3x/4x ud_near_1_30_12_2_short:1x/6x
    python -m backtesting.pairs ud_range ud_near limit --combo LONG SHORT LONG SHORT   # two pairs, summed first

Mirror pairs are left out: a short whose stops are the long's swapped ('b/a' against 'a/b') has the long's two levels,
so the pair is one trade taken both ways and earns only from the fill model and costs. The ranking is in sample (the
best of every stops pair); python -m backtesting.walk_forward tests picking from the past only.
--portfolio: from the best pair, add the pair of another condition (one per condition) that raises the portfolio's
Calmar most, while it rises by at least 0.02 (at most --max-pairs); picked on the whole period, 2026 included.
--holdout: everything is picked on the sessions before TEST_FROM (2026) only, the filters included (--min-trades,
--positive-legs), and the picks are then scored on the sessions from it: an out-of-sample test. Its files end in
_holdout, so the in-sample ones are kept. --singles (with --holdout): no pairs; per condition and side the run with the
best Calmar, the top of each side scored the same way.
Writes pairs.parquet and acc_pnl_pairs.png to the sweep folder; with --portfolio also portfolio.parquet,
acc_pnl_portfolio.png (the portfolio and each pair in one panel) and acc_pnl_portfolio_pairs.png (a panel per pair,
long and short split).
"""
import argparse
import datetime
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from backtesting.walk_forward import statistics
from params import RESULTS_DIR

TEST_FROM = datetime.date(2026, 1, 1)


def calmar(x: np.ndarray, years: float):
    """Columns of daily net -> (net per year / max drawdown, max drawdown), drawdown from a peak starting at 0."""
    cum = np.cumsum(x, axis=0)
    mdd = (np.maximum.accumulate(np.maximum(cum, 0.0), axis=0) - cum).max(axis=0)
    return cum[-1] / years / np.where(mdd > 0, mdd, np.nan), mdd


def martin(x: np.ndarray, years: float):
    """
    Columns of daily net -> (net per year / Ulcer index, Ulcer index): the Ulcer index is the root mean square of the
    drawdown ($, from a peak starting at 0) over every session, so every dip counts, deep and long ones most.
    """
    cum = np.cumsum(x, axis=0)
    ui = np.sqrt(((np.maximum.accumulate(np.maximum(cum, 0.0), axis=0) - cum) ** 2).mean(axis=0))
    no_dip = np.where(cum[-1] > 0, np.inf, np.nan)  # never below its peak: the best there is, if it made money
    return np.where(ui > 0, cum[-1] / years / np.where(ui > 0, ui, 1), no_dip), ui


def straightness(x: np.ndarray, years: float):
    """
    Columns of daily net -> (R2 of the accumulated net against a straight line, signed by the line's slope; slope in
    $ per session). 1 = a straight line up, saw-like or stepped curves lower, falling ones negative.
    """
    cum = np.cumsum(x, axis=0)
    s = (np.arange(len(cum)) - (len(cum) - 1) / 2)[:, None]
    mean = cum.mean(axis=0)
    slope = (s * (cum - mean)).sum(axis=0) / (s ** 2).sum()
    ss_tot = ((cum - mean) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1 - ((cum - mean - slope * s) ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1), np.nan)
    return np.sign(slope) * r2, slope


def sharpe(x: np.ndarray, years: float):
    """Columns of daily net (every session, 0 without a trade) -> (annualised Sharpe, daily standard deviation)."""
    sd = x.std(axis=0, ddof=1)
    return np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(252), np.nan), sd


def sortino(x: np.ndarray, years: float):
    """Columns of daily net -> (annualised Sortino: the mean over the downside deviation, only losing days counting as
    risk; downside deviation)."""
    down = np.sqrt((np.minimum(x, 0.0) ** 2).mean(axis=0))
    return np.where(down > 0, x.mean(axis=0) / np.where(down > 0, down, 1) * np.sqrt(252), np.nan), down


METRICS = {'calmar': calmar, 'martin': martin, 'r2': straightness, 'sharpe': sharpe, 'sortino': sortino}


class Runs:
    """The candidate runs (signal x stops, >= min_trades; positive_legs: only net positive ones; min_net: only those
    that made at least that many $ over t; min_year: only those that made at least that many $ in every calendar year
    of t, pro rata for a part year by its sessions / 252) and their daily net."""

    def __init__(self, t: pl.DataFrame, min_trades: int, positive_legs: bool = False, metric: str = 'calmar',
                 min_net: float | None = None, min_year: float | None = None):
        self.metric, self.score = metric, METRICS[metric]  # what best_short, best_pairs and portfolio rank by
        runs = (t.group_by('signal', 'stops', 'side').agg(pl.len().alias('trades'), pl.col('net_usd').sum().alias('net'))
                .filter((pl.col('trades') >= min_trades) & ((pl.col('net') > 0) | (not positive_legs))
                        & ((pl.col('net') >= min_net) | (min_net is None))))
        if min_year is not None:
            year = pl.col('date').dt.year().alias('year')
            days = t.group_by(year).agg(pl.col('date').n_unique().alias('days'))
            per_year = t.group_by('signal', 'stops', year).agg(pl.col('net_usd').sum().alias('year_net'))
            every = (runs.select('signal', 'stops').join(days, how='cross')  # a year without a trade counts as 0
                     .join(per_year, on=['signal', 'stops', 'year'], how='left').fill_null(0)
                     .group_by('signal', 'stops').agg((pl.col('year_net') >= min_year * pl.col('days') / 252).all().alias('ok')))
            runs = runs.join(every.filter('ok').select('signal', 'stops'), on=['signal', 'stops'])
        if runs.is_empty():
            raise ValueError('no run passes the filters (--min-trades, --positive-legs, --min-net, --min-year)')
        runs = (runs.sort('signal', 'stops').with_row_index('k')
                .with_columns(cond=pl.col('signal').str.replace(r'_(long|short|reversed)$', '')))
        t = t.join(runs.select('signal', 'stops', 'k'), on=['signal', 'stops'])
        self.dates = t['date'].unique().sort()
        g = t.group_by('k', 'date').agg(pl.col('net_usd').sum())
        self.D = np.zeros((len(self.dates), len(runs)))
        self.D[self.dates.search_sorted(g['date']).to_numpy(), g['k'].to_numpy()] = g['net_usd'].to_numpy()
        self.test = (self.dates >= TEST_FROM).to_numpy()
        self.yrs_all = (self.dates.max() - self.dates.min()).days / 365.25
        self.yrs_in = (self.dates.filter(~pl.Series(self.test)).max() - self.dates.min()).days / 365.25
        self.signal = runs['signal'].to_list()
        self.side, self.cond = runs['side'].to_numpy(), runs['cond'].to_numpy()
        self.stops = np.array(runs['stops'].to_list())
        self.swapped = np.array(['/'.join(st.split('/')[::-1]) for st in self.stops])

    def best_short(self, i: int, base: np.ndarray):
        """For long run i: (score, short run) of the best non-mirror short of its condition by the metric, with `base`
        (daily net already held) added; (nan, None) when there is none."""
        shorts = np.flatnonzero((self.cond == self.cond[i]) & (self.side == -1))
        ca, _ = self.score(base[:, None] + self.D[:, [i]] + self.D[:, shorts], self.yrs_all)
        ca = np.where(self.stops[shorts] == self.swapped[i], np.nan, ca)  # the long's mirror: not a pair
        if np.isnan(ca).all():
            return np.nan, None
        j = int(np.nanargmax(ca))
        return ca[j], int(shorts[j])

    def row(self, x: np.ndarray) -> dict:
        """Calmar, Martin, Ulcer index, R2 (straightness), max drawdown, net per year, Calmar before 2026 and 2026 net of a
        daily net."""
        ca, ma = calmar(x[:, None], self.yrs_all)
        ci, _ = calmar(x[~self.test][:, None], self.yrs_in)
        (mt,), (ui,) = martin(x[:, None], self.yrs_all)
        return {'calmar': float(ca[0]), 'martin': float(mt), 'ulcer': float(ui),
                'sharpe': float(sharpe(x[:, None], self.yrs_all)[0][0]), 'sortino': float(sortino(x[:, None], self.yrs_all)[0][0]),
                'r2': float(straightness(x[:, None], self.yrs_all)[0][0]), 'max_dd': float(ma[0]),
                'net_per_year': float(x.sum() / self.yrs_all), 'calmar_before_2026': float(ci[0]),
                'net_2026': float(x[self.test].sum())}

    def names(self, i: int, j: int) -> dict:
        return {'condition': self.cond[i], 'long_signal': self.signal[i], 'long_stops': self.stops[i],
                'short_signal': self.signal[j], 'short_stops': self.stops[j]}


def best_pairs(r: Runs) -> pl.DataFrame:
    """The best non-mirror long x short pair of every condition, by the metric over the whole period."""
    rows, zero = [], np.zeros(len(r.dates))
    for i in np.flatnonzero(r.side == 1):
        _, j = r.best_short(int(i), zero)
        if j is not None:
            rows.append({**r.names(int(i), j), **r.row(r.D[:, i] + r.D[:, j])})
    return (pl.DataFrame(rows).sort(r.metric, descending=True, nulls_last=True)
            .group_by('condition', maintain_order=True).first())


def portfolio(r: Runs, max_pairs: int, min_gain: float = 0.02) -> pl.DataFrame:
    """From the best pair, add the pair of another condition that raises the metric most, while it rises by min_gain."""
    base, used, steps = np.zeros(len(r.dates)), set(), []
    while len(steps) < max_pairs:
        best = (-np.inf, None)
        for i in np.flatnonzero((r.side == 1) & ~np.isin(r.cond, list(used))):
            ca, j = r.best_short(int(i), base)
            if j is not None and ca > best[0]:
                best = (ca, (int(i), j))
        if best[1] is None or (steps and best[0] < steps[-1][r.metric] + min_gain):
            break
        i, j = best[1]
        base = base + r.D[:, i] + r.D[:, j]
        used.add(r.cond[i])
        steps.append({'pairs': len(steps) + 1, **r.names(i, j), **r.row(base)})
    return pl.DataFrame(steps)


def pair_trades(t: pl.DataFrame, r: dict) -> pl.DataFrame:
    """Both legs' trades of one pair row."""
    return pl.concat([t.filter((pl.col('signal') == r['long_signal']) & (pl.col('stops') == r['long_stops'])),
                      t.filter((pl.col('signal') == r['short_signal']) & (pl.col('stops') == r['short_stops']))])


def breakdown(trades: pl.DataFrame) -> pl.DataFrame:
    """Net per trade by session block (shifted clock: Asia 00-08, Europe 08-16, US 16-23), regular hours, `near`."""
    minute = pl.col('entry_ts').dt.hour().cast(pl.Int32) * 60 + pl.col('entry_ts').dt.minute().cast(pl.Int32)
    t = trades.with_columns(
        session=pl.when(minute < 8 * 60).then(pl.lit('Asia')).when(minute < 16 * 60).then(pl.lit('Europe'))
        .otherwise(pl.lit('US')),
        rth=pl.when((minute >= 15 * 60 + 30) & (minute < 22 * 60)).then(pl.lit('RTH')).otherwise(pl.lit('outside RTH')))
    labels = ['session', 'rth'] + (['near_ranges'] if 'near' in t.columns else [])
    if 'near' in t.columns:
        t = t.with_columns(near_ranges=pl.when(pl.col('near').str.contains(',')).then(pl.lit('2+ ranges'))
                           .otherwise(pl.lit('1 range')))
    return pl.concat([t.group_by(label).agg(pl.lit(label).alias('label'), pl.len().alias('trades'),
                                             pl.col('net_usd').mean().round(1).alias('net_per_trade'),
                                             pl.col('net_usd').sum().round(0).alias('net'))
                      .rename({label: 'value'}) for label in labels]).select('label', 'value', 'trades', 'net_per_trade', 'net')


def plot(t: pl.DataFrame, pairs: pl.DataFrame, dates: pl.Series, out_path: str, together: str | None = None,
         metric: str = 'calmar'):
    """
    One panel per pair: its accumulated net (drawdown shaded, 2026 in another colour), its long leg and its short leg,
    drawn on the days it traded. together: a title for a first panel with every pair summed (the portfolio), all its
    longs and all its shorts; a `size` column (contracts per leg, default 1) weights each pair in it. metric 'martin':
    titled by Martin and the Ulcer index to 2025 and in 2026 (else by Calmar).
    """
    x = dates.to_numpy()
    test = x >= np.datetime64(TEST_FROM)
    yrs_all = (dates.max() - dates.min()).days / 365.25
    yrs_in = (dates.filter(~pl.Series(test)).max() - dates.min()).days / 365.25
    yrs_26 = max((dates.max() - dates.filter(pl.Series(test)).min()).days, 1) / 365.25 if test.any() else 1

    def daily(signal, stops):
        """The run's net and trade count per session (0 on sessions without a trade)."""
        g = (t.filter((pl.col('signal') == signal) & (pl.col('stops') == stops))
             .group_by('date').agg(pl.col('net_usd').sum(), pl.len().alias('n')))
        d = pl.DataFrame({'date': dates}).join(g, on='date', how='left').fill_null(0).sort('date')
        return d['net_usd'].to_numpy(), d['n'].to_numpy()

    def title(name, net):
        (ca,), (mdd,) = calmar(net[:, None], yrs_all)
        (ci,), _ = calmar(net[~test][:, None], yrs_in)
        (_,), (mdd26,) = calmar(net[test][:, None], 1)
        if metric == 'martin':
            (mi,), (uii,) = martin(net[~test][:, None], yrs_in)
            (m26,), (ui26,) = martin(net[test][:, None], yrs_26)
            return ((f"{name}: " if name else '') + f"to 2025: Martin {mi:.2f}, Ulcer index \\${uii:,.0f}, "
                    f"\\${net[~test].sum() / yrs_in:,.0f}/year; 2026: Martin {m26:.2f}, Ulcer index \\${ui26:,.0f}, "
                    f"net \\${net[test].sum():,.0f}")
        return ((f"{name}: " if name else '') + f"Calmar {ca:.2f} (before 2026 {ci:.2f}), max DD \\${mdd:,.0f}, "
                f"\\${net.sum() / yrs_all:,.0f}/year; 2026 net \\${net[test].sum():,.0f}, max DD \\${mdd26:,.0f}")

    legs = [(daily(r['long_signal'], r['long_stops']), daily(r['short_signal'], r['short_stops']))
            for r in pairs.iter_rows(named=True)]
    # A pair panel is titled by its numbers; the legend names its buy (long) and sell (short) legs
    panels = [(None, a, b, f"buy {r['long_signal']} {r['long_stops']}", f"sell {r['short_signal']} {r['short_stops']}")
              for r, (a, b) in zip(pairs.iter_rows(named=True), legs)]
    if together:
        size = pairs['size'].to_list() if 'size' in pairs.columns else [1] * len(legs)
        panels.insert(0, (together, (sum(n * a[0] for n, (a, _) in zip(size, legs)), sum(a[1] for a, _ in legs)),
                          (sum(n * b[0] for n, (_, b) in zip(size, legs)), sum(b[1] for _, b in legs)),
                          'all longs', 'all shorts'))
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.2 * len(panels)), squeeze=False)
    for ax, (name, (a, na), (b, nb), long_label, short_label) in zip(axes[:, 0], panels):
        on = (na + nb) > 0  # the days it traded: the curves join them, as the drawdown and Calmar see them
        xs, both = x[on], np.cumsum(a + b)[on]
        k = int(test[on].argmax()) if test[on].any() else len(xs)
        ax.fill_between(xs, both, np.maximum.accumulate(np.maximum(both, 0.0)), color='#d62728', alpha=0.15, lw=0)
        ax.plot(xs, np.cumsum(a)[on], color='#2ca02c', lw=0.9, alpha=0.8, label=long_label)
        ax.plot(xs, np.cumsum(b)[on], color='#9467bd', lw=0.9, alpha=0.8, label=short_label)
        ax.plot(xs[:k + 1], both[:k + 1], color='#2a6fdb', lw=2, label='combined to 2025')
        ax.plot(xs[k:], both[k:], color='#e8762b', lw=2, label='combined 2026')
        ax.axhline(0, color='#999', lw=0.8)
        ax.set_title(title(name, a + b), fontsize=10, loc='left')
        ax.set_ylabel('acc net $')
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left', frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def plot_ud(t: pl.DataFrame, pairs: pl.DataFrame, dates: pl.Series, out_path: str):
    """
    The UD studies' look: one panel per pair on every session, titled with its trades and win rate; the legend gives
    each leg's side and stops.
    """
    x = dates.to_numpy()
    k = int((x >= np.datetime64(TEST_FROM)).argmax())

    def leg(signal, stops):
        sel = t.filter((pl.col('signal') == signal) & (pl.col('stops') == stops))
        g = sel.group_by('date').agg(pl.col('net_usd').sum())
        daily = pl.DataFrame({'date': dates}).join(g, on='date', how='left').fill_null(0).sort('date')['net_usd'].to_numpy()
        return daily, len(sel), int((sel['net_usd'] > 0).sum())

    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 3.1 * len(pairs)), squeeze=False)
    for ax, r in zip(axes[:, 0], pairs.iter_rows(named=True)):
        (a, na, wa), (b, nb, wb) = leg(r['long_signal'], r['long_stops']), leg(r['short_signal'], r['short_stops'])
        both = np.cumsum(a + b)
        ax.fill_between(x, both, np.maximum.accumulate(np.maximum(both, 0.0)), color='#d62728', alpha=0.15, lw=0)
        ax.plot(x, np.cumsum(a), color='#2ca02c', lw=0.9, alpha=0.8, label=f"buy {r['long_stops']}")
        ax.plot(x, np.cumsum(b), color='#9467bd', lw=0.9, alpha=0.8, label=f"sell {r['short_stops']}")
        ax.plot(x[:k + 1], both[:k + 1], color='#2a6fdb', lw=2, label='pair, to 2025')
        ax.plot(x[k:], both[k:], color='#e8762b', lw=2, label='pair, 2026')
        ax.axhline(0, color='#999', lw=0.8)
        ax.set_title(f"{r['condition']}: {na + nb} trades, win {(wa + wb) / (na + nb):.0%}, Calmar {r['calmar']:.2f} "
                     f"(before 2026 {r['calmar_before_2026']:.2f}), max DD \\${r['max_dd']:,.0f}, "
                     f"\\${r['net_per_year']:,.0f}/year; 2026 \\${r['net_2026']:,.0f}", fontsize=9, loc='left')
        ax.set_ylabel('acc net $')
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left', frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def plot_overview(t: pl.DataFrame, steps: pl.DataFrame, dates: pl.Series, out_path: str, metric: str = 'calmar'):
    """The portfolio in one panel: its accumulated net (drawdown shaded, 2026 in another colour) and each pair's.
    metric 'martin': titled by Martin and the Ulcer index to 2025 and in 2026 (else by Calmar)."""
    x = dates.to_numpy()
    test = x >= np.datetime64(TEST_FROM)
    k = int(test.argmax())
    yrs_all = (dates.max() - dates.min()).days / 365.25
    yrs_in = (dates.filter(~pl.Series(test)).max() - dates.min()).days / 365.25

    def daily(signal, stops):
        g = t.filter((pl.col('signal') == signal) & (pl.col('stops') == stops)).group_by('date').agg(pl.col('net_usd').sum())
        return pl.DataFrame({'date': dates}).join(g, on='date', how='left').fill_null(0).sort('date')['net_usd'].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    total = np.zeros(len(x))
    for r in steps.iter_rows(named=True):
        pair = daily(r['long_signal'], r['long_stops']) + daily(r['short_signal'], r['short_stops'])
        total += pair
        ax.plot(x, np.cumsum(pair), lw=0.9, alpha=0.8,
                label=f"buy {r['long_signal']} {r['long_stops']} + sell {r['short_signal']} {r['short_stops']}")
    cum = np.cumsum(total)
    ax.fill_between(x, cum, np.maximum.accumulate(np.maximum(cum, 0.0)), color='#d62728', alpha=0.15, lw=0)
    ax.plot(x[:k + 1], cum[:k + 1], color='#2a6fdb', lw=2.2, label='portfolio to 2025')
    ax.plot(x[k:], cum[k:], color='#e8762b', lw=2.2, label='portfolio 2026')
    ax.axhline(0, color='#999', lw=0.8)
    (ca,), (mdd,) = calmar(total[:, None], yrs_all)
    (ci,), _ = calmar(total[~test][:, None], yrs_in)
    (_,), (mdd26,) = calmar(total[test][:, None], 1)
    if metric == 'martin':
        yrs_26 = max((dates.max() - dates.filter(pl.Series(test)).min()).days, 1) / 365.25
        (mi,), (uii,) = martin(total[~test][:, None], yrs_in)
        (m26,), (ui26,) = martin(total[test][:, None], yrs_26)
        ax.set_title(f'{len(steps)} same-condition pairs, 1 contract per leg: to 2025 Martin {mi:.2f}, Ulcer index '
                     f'\\${uii:,.0f}, \\${total[~test].sum() / yrs_in:,.0f}/year\n2026: Martin {m26:.2f}, Ulcer index '
                     f'\\${ui26:,.0f}, net \\${total[test].sum():,.0f}', fontsize=9.5, loc='left')
    else:
        ax.set_title(f'{len(steps)} same-condition pairs, 1 contract per leg: Calmar {ca:.2f} over the whole period (max DD '
                     f'\\${mdd:,.0f}), \\${total.sum() / yrs_all:,.0f}/year\nbefore 2026 {ci:.2f}; 2026 net '
                     f'\\${total[test].sum():,.0f}, max DD \\${mdd26:,.0f}', fontsize=9.5, loc='left')
    ax.set_ylabel('acc net $')
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def holdout_scores(t: pl.DataFrame, dates: pl.Series, rows: pl.DataFrame, cumulative: bool) -> pl.DataFrame:
    """
    The picks on the test sessions (from TEST_FROM): trades, win rate, net, net per year and max drawdown, per pair or,
    with `cumulative`, of the portfolio so far at each step.
    """
    test_dates = dates.filter(dates >= TEST_FROM)
    years = max((test_dates.max() - test_dates.min()).days, 1) / 365.25
    total, out = np.zeros(len(test_dates)), []
    held = []
    for r in rows.iter_rows(named=True):
        trades = pair_trades(t, r).filter(pl.col('date') >= TEST_FROM)
        x = (pl.DataFrame({'date': test_dates}).join(trades.group_by('date').agg(pl.col('net_usd').sum()), on='date',
                                                     how='left').fill_null(0).sort('date')['net_usd'].to_numpy())
        held.append(trades)
        if cumulative:
            total, trades = total + x, pl.concat(held)
        else:
            total = x
        _, (mdd,) = calmar(total[:, None], years)
        (ma,), (ui,) = martin(total[:, None], years)
        out.append({'test_trades': len(trades), 'test_win': float((trades['net_usd'] > 0).mean()) if len(trades) else None,
                    'test_net': float(total.sum()), 'test_net_per_year': float(total.sum() / years),
                    'test_max_dd': float(mdd), 'test_martin': float(ma), 'test_ulcer': float(ui)})
    return pl.DataFrame(out)


def print_statistics(t: pl.DataFrame, pairs: pl.DataFrame, dates: pl.Series, breakdowns: bool):
    """Each pair's and each leg's statistics (walk_forward.statistics, plus Martin and the Ulcer index to 2025 and
    in 2026), and optionally its label breakdown."""
    rows, test = [], (dates >= TEST_FROM).to_numpy()
    yrs_in = (dates.filter(~pl.Series(test)).max() - dates.min()).days / 365.25
    yrs_26 = max((dates.max() - dates.filter(pl.Series(test)).min()).days, 1) / 365.25 if test.any() else 1
    for n, r in enumerate(pairs.iter_rows(named=True), 1):
        both = pair_trades(t, r)
        for part, sel in (('pair', both), (f"long: buy {r['long_stops']}", both.filter(pl.col('side') == 1)),
                          (f"short: sell {r['short_stops']}", both.filter(pl.col('side') == -1))):
            x = (pl.DataFrame({'date': dates}).join(sel.group_by('date').agg(pl.col('net_usd').sum()), on='date',
                                                    how='left').fill_null(0).sort('date')['net_usd'].to_numpy())
            (mi,), (uii,) = martin(x[~test][:, None], yrs_in)
            (m26,), (ui26,) = martin(x[test][:, None], yrs_26)
            rows.append({'#': n, 'condition': r['condition'], 'part': part, **statistics(sel, dates),
                         'martin_to_2025': round(float(mi), 2), 'ulcer_to_2025': round(float(uii)),
                         'martin_2026': round(float(m26), 2), 'ulcer_2026': round(float(ui26))})
        if breakdowns:
            print(f"\n#{n} {r['condition']}: net per trade by label")
            print(breakdown(both))
    stats = pl.DataFrame(rows)
    print('\nEach pair and each leg:')
    print(stats.select('#', 'condition', 'part', 'trades', 'win_rate', 'avg_win', 'avg_loss', 'profit_factor', 'net_per_year'))
    print(stats.select('#', 'condition', 'part', 'martin_to_2025', 'ulcer_to_2025', 'martin_2026', 'ulcer_2026',
                       'net_2026', 'max_dd', 'calmar', 'sharpe', 'positive_months'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('sweep', nargs='+', help='the sweep folder: a name under results/sweep/ (e.g. ud_range) or a '
                                                 'path; several: their trades together, the outputs in the first')
    parser.add_argument('--top', type=int, default=5, help='pairs plotted and detailed')
    parser.add_argument('--min-trades', type=int, default=100, help='per run over the whole period')
    parser.add_argument('--breakdown', action='store_true', help='net per trade by session, regular hours and near')
    parser.add_argument('--positive-legs', action='store_true', help='only legs net positive over the whole period')
    parser.add_argument('--min-net', type=float, default=None,
                        help='only runs that made at least this many $ (with --holdout: before 2026)')
    parser.add_argument('--min-year', type=float, default=None,
                        help='only runs that made at least this many $ in every year (a part year pro rata)')
    parser.add_argument('--portfolio', action='store_true', help='also build the best-Calmar portfolio of pairs')
    parser.add_argument('--max-pairs', type=int, default=8, help='most pairs in the portfolio')
    parser.add_argument('--look', choices=('rsi', 'ud'), default='rsi',
                        help="the top pairs plot's look: rsi (drawn on the days a pair traded) or ud (every session)")
    parser.add_argument('--holdout', action='store_true', help='pick on the sessions before 2026 only, test on 2026')
    parser.add_argument('--metric', choices=tuple(METRICS), default='calmar',
                        help='rank by calmar (worst drawdown), martin (every drawdown: Ulcer index) or r2 (a straight line up)')
    parser.add_argument('--singles', action='store_true', help='with --holdout: single runs per side, no pairs')
    parser.add_argument('--only', default=None, help='only the signals matching this regex (e.g. one distance)')
    parser.add_argument('--tag', default=None, help='a suffix for the output file names (keeps the other runs)')
    parser.add_argument('--combo', nargs='+', metavar='SIGNAL:STOPS', default=None,
                        help='plot a long run and a short run together (any conditions), with their statistics; '
                             'several pairs (LONG SHORT LONG SHORT ...) are also summed in a first panel')
    args = parser.parse_args(argv)
    folders = [f if os.path.isdir(f) else os.path.join(RESULTS_DIR, 'sweep', f) for f in args.sweep]
    folder = folders[0]
    t = (pl.concat([pl.read_parquet(f) for d in folders for f in sorted(glob.glob(os.path.join(d, 'trades_*.parquet')))],
                   how='diagonal_relaxed').with_columns(date=pl.col('entry_ts').dt.date()))
    if args.only:
        t = t.filter(pl.col('signal').str.contains(args.only))
    if args.combo:
        legs = [c.split(':', 1) for c in args.combo]
        if len(legs) % 2:
            parser.error('--combo takes pairs: LONG SHORT [LONG SHORT ...]')
        row = pl.DataFrame({'condition': [f'{lg[0]} + {sh[0]}' for lg, sh in zip(legs[::2], legs[1::2])],
                            'long_signal': [lg[0] for lg in legs[::2]], 'long_stops': [lg[1] for lg in legs[::2]],
                            'short_signal': [sh[0] for sh in legs[1::2]], 'short_stops': [sh[1] for sh in legs[1::2]]})
        # A pair given more than once trades that many contracts per leg in the total (its own panel shows one)
        row = row.group_by(row.columns, maintain_order=True).agg(pl.len().cast(pl.Int64).alias('size'))
        dates = t['date'].unique().sort()
        out = os.path.join(folder, f"acc_pnl_combo{f'_{args.tag}' if args.tag else ''}.png")
        sizes = ' + '.join(f"{n} x pair {i}" for i, n in enumerate(row['size'], 1)) if row['size'].max() > 1 else '1 contract per leg'
        together = f'{row.height} pairs together ({sizes})' if row.height > 1 else None
        plot(t, row, dates, out, together=together, metric=args.metric)
        pl.Config.set_tbl_rows(40).set_tbl_width_chars(250).set_tbl_hide_dataframe_shape(True).set_fmt_str_lengths(40)
        pl.Config.set_tbl_cols(20)
        print_statistics(t, row, dates, args.breakdown)
        print(out)
        return
    if args.holdout:
        (holdout_singles if args.singles else holdout)(t, folder, args)
        return
    runs = Runs(t, args.min_trades, args.positive_legs, args.metric, args.min_net, args.min_year)
    pairs = best_pairs(runs)
    pairs.write_parquet(os.path.join(folder, 'pairs.parquet'))
    top = pairs.head(args.top)
    (plot_ud if args.look == 'ud' else plot)(t, top, runs.dates, os.path.join(folder, 'acc_pnl_pairs.png'))

    pl.Config.set_tbl_rows(40).set_tbl_width_chars(250).set_tbl_hide_dataframe_shape(True).set_fmt_str_lengths(40)
    pl.Config.set_tbl_cols(20)
    rounded = [pl.col('calmar', 'calmar_before_2026').round(2), pl.col('max_dd', 'net_per_year', 'net_2026').round(0)]
    print(f'{pairs.height} conditions with a pair; best pair per condition by Calmar:')
    print(pairs.select('condition', 'long_stops', 'short_stops', 'calmar', 'max_dd', 'net_per_year', 'calmar_before_2026',
                       'net_2026').with_columns(*rounded).head(20))
    print_statistics(t, top, runs.dates, args.breakdown)
    print(os.path.join(folder, 'acc_pnl_pairs.png'))

    if args.portfolio:
        steps = portfolio(runs, args.max_pairs)
        steps.write_parquet(os.path.join(folder, 'portfolio.parquet'))
        out = os.path.join(folder, 'acc_pnl_portfolio.png')
        plot_overview(t, steps, runs.dates, out)
        plot(t, steps, runs.dates, os.path.join(folder, 'acc_pnl_portfolio_pairs.png'),
             together=f'{len(steps)} pairs together (1 contract per leg)')
        print('\nPortfolio: each step adds the pair that raises the Calmar most (the numbers are the portfolio so far):')
        print(steps.select('pairs', 'condition', 'long_stops', 'short_stops', 'calmar', 'max_dd', 'net_per_year',
                           'calmar_before_2026', 'net_2026').with_columns(*rounded))
        print_statistics(t, steps, runs.dates, args.breakdown)
        print(out)


def holdout(t: pl.DataFrame, folder: str, args):
    """--holdout: pairs and portfolio picked on the sessions before TEST_FROM, scored on the sessions from it."""
    runs = Runs(t.filter(pl.col('date') < TEST_FROM), args.min_trades, args.positive_legs, args.metric,  # 2026 unseen
                args.min_net, args.min_year)
    dates = t['date'].unique().sort()
    pl.Config.set_tbl_rows(40).set_tbl_width_chars(250).set_tbl_hide_dataframe_shape(True).set_fmt_str_lengths(40)
    pl.Config.set_tbl_cols(20)
    picked = {'calmar': 'pick_calmar', 'martin': 'pick_martin', 'ulcer': 'pick_ulcer', 'r2': 'pick_r2', 'sharpe': 'pick_sharpe',
              'sortino': 'pick_sortino', 'max_dd': 'pick_max_dd',
              'net_per_year': 'pick_net_per_year'}
    show = ['pick_calmar', 'pick_martin', 'pick_ulcer', 'pick_r2', 'pick_max_dd', 'pick_net_per_year', 'test_trades',
            'test_win', 'test_net', 'test_net_per_year', 'test_martin', 'test_ulcer', 'test_max_dd']
    rounded = [pl.col('pick_calmar', 'pick_martin', 'pick_r2', 'test_win', 'test_martin').round(2),
               pl.col('pick_ulcer', 'pick_max_dd', 'pick_net_per_year', 'test_net', 'test_net_per_year', 'test_ulcer',
                      'test_max_dd').round(0)]

    pairs = best_pairs(runs).drop('calmar_before_2026', 'net_2026').rename(picked)
    top = pairs.head(args.top)
    scored = pl.concat([top, holdout_scores(t, dates, top, cumulative=False)], how='horizontal_extend')
    tag = f'_{args.tag}' if args.tag else ''
    scored.write_parquet(os.path.join(folder, f'pairs_holdout{tag}.parquet'))
    out = os.path.join(folder, f'acc_pnl_pairs_holdout{tag}.png')
    plot(t, top, dates, out, metric=args.metric)
    print(f'Picked on {runs.dates.min()} .. {runs.dates.max()} only; tested from {TEST_FROM} (2026 not seen by the pick).')
    print(f'The best pair per condition by {args.metric} before 2026, top {len(top)}, and how it did in 2026:')
    print(scored.select('condition', 'long_stops', 'short_stops', *show).with_columns(*rounded))
    print(out)
    if args.portfolio:
        steps = portfolio(runs, args.max_pairs).drop('calmar_before_2026', 'net_2026').rename(picked)
        scored = pl.concat([steps, holdout_scores(t, dates, steps, cumulative=True)], how='horizontal_extend')
        scored.write_parquet(os.path.join(folder, f'portfolio_holdout{tag}.parquet'))
        out = os.path.join(folder, f'acc_pnl_portfolio_holdout{tag}.png')
        plot_overview(t, steps, dates, out, metric=args.metric)
        plot(t, steps, dates, os.path.join(folder, f'acc_pnl_portfolio_pairs_holdout{tag}.png'),
             together=f'{len(steps)} pairs picked before 2026 (1 contract per leg)', metric=args.metric)
        print(f'\nPortfolio picked before 2026 (each step adds the pair that raises its pre-2026 {args.metric} most), and the '
              'portfolio so far in 2026:')
        print(scored.select('pairs', 'condition', 'long_stops', 'short_stops', *show).with_columns(*rounded))
        print(out)


def holdout_singles(t: pl.DataFrame, folder: str, args):
    """--holdout --singles: per condition and side the best run by Calmar before TEST_FROM, top of each side on 2026."""
    runs = Runs(t.filter(pl.col('date') < TEST_FROM), args.min_trades, args.positive_legs, args.metric,  # 2026 unseen
                args.min_net, args.min_year)
    dates = t['date'].unique().sort()
    rows = [{'condition': runs.cond[k], 'signal': runs.signal[k], 'stops': runs.stops[k], 'side': int(runs.side[k]),
             **runs.row(runs.D[:, k])} for k in range(len(runs.signal))]
    best = (pl.DataFrame(rows).drop('calmar_before_2026', 'net_2026')
            .rename({'calmar': 'pick_calmar', 'martin': 'pick_martin', 'ulcer': 'pick_ulcer', 'r2': 'pick_r2',
                     'sharpe': 'pick_sharpe', 'sortino': 'pick_sortino', 'max_dd': 'pick_max_dd',
                     'net_per_year': 'pick_net_per_year'})
            .sort(f'pick_{args.metric}', descending=True, nulls_last=True)
            .group_by('condition', 'side', maintain_order=True).first())
    top = pl.concat([best.filter(pl.col('side') == sd).head(args.top) for sd in (1, -1)])
    test_dates = dates.filter(dates >= TEST_FROM)
    years = max((test_dates.max() - test_dates.min()).days, 1) / 365.25
    scores, nets = [], []
    for r in top.iter_rows(named=True):
        tr = t.filter((pl.col('signal') == r['signal']) & (pl.col('stops') == r['stops']))
        x = (pl.DataFrame({'date': dates}).join(tr.group_by('date').agg(pl.col('net_usd').sum()), on='date', how='left')
             .fill_null(0).sort('date')['net_usd'].to_numpy())
        nets.append(x)
        test, test_x = tr.filter(pl.col('date') >= TEST_FROM), x[(dates >= TEST_FROM).to_numpy()]
        _, (mdd,) = calmar(test_x[:, None], years)
        (ma,), (ui,) = martin(test_x[:, None], years)
        scores.append({'test_trades': len(test), 'test_win': float((test['net_usd'] > 0).mean()) if len(test) else None,
                       'test_net': float(test_x.sum()), 'test_net_per_year': float(test_x.sum() / years),
                       'test_max_dd': float(mdd), 'test_martin': float(ma), 'test_ulcer': float(ui),
                       'test_sharpe': float(sharpe(test_x[:, None], years)[0][0]),
                       'test_sortino': float(sortino(test_x[:, None], years)[0][0])})
    scored = pl.concat([top, pl.DataFrame(scores)], how='horizontal_extend')
    tag = f'_{args.tag}' if args.tag else ''
    scored.write_parquet(os.path.join(folder, f'singles_holdout{tag}.parquet'))

    x = dates.to_numpy()
    k = int((x >= np.datetime64(TEST_FROM)).argmax())
    fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.6 * len(top)), squeeze=False)
    for ax, r, net in zip(axes[:, 0], scored.iter_rows(named=True), nets):
        cum = np.cumsum(net)
        ax.fill_between(x, cum, np.maximum.accumulate(np.maximum(cum, 0.0)), color='#d62728', alpha=0.15, lw=0)
        ax.plot(x[:k + 1], cum[:k + 1], color='#2a6fdb', lw=1.8, label='picked on (to 2025)')
        ax.plot(x[k:], cum[k:], color='#e8762b', lw=1.8, label='test (2026)')
        ax.axhline(0, color='#999', lw=0.8)
        if args.metric == 'martin':
            numbers = (f"to 2025: Martin {r['pick_martin']:.2f}, Ulcer \\${r['pick_ulcer']:,.0f}, "
                       f"\\${r['pick_net_per_year']:,.0f}/year; 2026: {r['test_trades']} trades, Martin "
                       f"{r['test_martin']:.2f}, Ulcer \\${r['test_ulcer']:,.0f}, net \\${r['test_net']:,.0f}")
        elif args.metric in ('sharpe', 'sortino'):
            name = args.metric.capitalize()
            numbers = (f"to 2025: {name} {r[f'pick_{args.metric}']:.2f}, max DD \\${r['pick_max_dd']:,.0f}, "
                       f"\\${r['pick_net_per_year']:,.0f}/year; 2026: {r['test_trades']} trades, {name} "
                       f"{r[f'test_{args.metric}']:.2f}, net \\${r['test_net']:,.0f}, max DD \\${r['test_max_dd']:,.0f}")
        else:
            numbers = (f"picked on Calmar {r['pick_calmar']:.2f}, Martin {r['pick_martin']:.2f}, R2 {r['pick_r2']:.2f}, "
                       f"\\${r['pick_net_per_year']:,.0f}/year; 2026: {r['test_trades']} trades, net "
                       f"\\${r['test_net']:,.0f}, max DD \\${r['test_max_dd']:,.0f}")
        ax.set_title(f"{'buy' if r['side'] == 1 else 'sell'} {r['signal']} {r['stops']}: {numbers}", fontsize=9, loc='left')
        ax.set_ylabel('acc net $')
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(loc='upper left', frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(folder, f'acc_pnl_singles_holdout{tag}.png')
    fig.savefig(out, dpi=110)

    pl.Config.set_tbl_rows(40).set_tbl_width_chars(250).set_tbl_hide_dataframe_shape(True).set_fmt_str_lengths(40)
    pl.Config.set_tbl_cols(20)
    print(f'Picked on {runs.dates.min()} .. {runs.dates.max()} only; tested from {TEST_FROM} (2026 not seen by the pick).')
    print(f'Per side, the best run of each condition by {args.metric} before 2026, top {args.top}, and how it did in 2026:')
    print(scored.select('side', 'signal', 'stops', 'pick_calmar', 'pick_martin', 'pick_ulcer', 'pick_r2', 'pick_sharpe',
                        'pick_sortino', 'pick_max_dd', 'pick_net_per_year', 'test_trades', 'test_win', 'test_net',
                        'test_net_per_year', 'test_martin', 'test_ulcer', 'test_sharpe', 'test_sortino', 'test_max_dd')
          .with_columns(pl.col('pick_calmar', 'pick_martin', 'pick_r2', 'pick_sharpe', 'pick_sortino', 'test_win',
                               'test_martin', 'test_sharpe', 'test_sortino').round(2),
                        pl.col('pick_ulcer', 'pick_max_dd', 'pick_net_per_year', 'test_net', 'test_net_per_year', 'test_ulcer',
                               'test_max_dd').round(0)))
    print(out)


if __name__ == '__main__':
    main()
