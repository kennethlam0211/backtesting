"""
Downstream of the backtest: queries its one table of trades (params.TRADES_PATH, every strategy x TP / SL pair) for
the leaderboard and the label analysis (where the net edge sits: hour, session, regular hours, news, weekday, side,
volatility, and each label crossed with side; and whether it holds up), then reports the final run with its
statistics and charts.

    python -m backtesting.backtest && python -m backtesting.analysis
        results/analysis/leaderboard.csv     every strategy x TP / SL: summary statistics, best total net first
        results/analysis/by_<label>.csv      per strategy x TP / SL x label value (by_hour, by_hour_x_side, ...)
        results/analysis/pooled.csv          per strategy x label value, the TP / SL pairs pooled
        results/analysis/*.png               hours, labels, stability
        results/final/                       the final run (config.FINAL, else the best net): statistics and charts

For queries of your own, load_trades() returns the trades with weekday, year and the volatility label added.

A slice is one strategy x TP/SL pair x label value (e.g. rsi_15_85, TP 40 SL 24, hour 13). Per slice: trades, gross
and net $ per trade, win rate, the t-stat of the net per trade, the net before / from config.SPLIT and the years it
was net positive. Thousands of slices are tested, so some look good by chance; the summary compares what passes with
what chance alone would give, and checks whether slices picked before SPLIT stay positive after it.
"""
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import Normalize, TwoSlopeNorm

from backtesting.backtest import LABEL_COLS, read_training_data
from backtesting.config import FINAL, LABEL_FEATURES, SPLIT
from backtesting.plots import (
    BASELINE,
    CATEGORICAL,
    DIVERGING,
    NEGATIVE,
    POSITIVE,
    TEXT_MUT,
    TEXT_PRI,
    TEXT_SEC,
    USD,
    BacktestPlots,
    _save,
    _style,
    _usd,
)
from backtesting.stats import BacktestStats
from params import RESULTS_DIR, TRADES_PATH, TRAINING_DATA_PATH

# Single labels, then each label crossed with the side
LABELS = [*LABEL_COLS, *[f'{c}_q' for c in LABEL_FEATURES], 'weekday', 'side']
SLICES = [[c] for c in LABELS] + [[c, 'side'] for c in LABELS if c != 'side']
MIN_TRADES = 50  # per half, for a slice to be judged
WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']


def load_trades(path=TRADES_PATH) -> pl.DataFrame:
    """
    The backtest's trades (every strategy x TP / SL pair, one table) plus `weekday` (1 Mon .. 5 Fri), `year` and the
    config.LABEL_FEATURES at the signal bar (raw and as quantile `_q`). Labels and side as Int64.
    """
    trades = pl.read_parquet(path).with_columns(
        pl.col(LABEL_COLS).cast(pl.Int64), side=pl.col('side').cast(pl.Int64),
        weekday=pl.col('date').dt.weekday().cast(pl.Int64), year=pl.col('date').dt.year(),
    )
    # Features at the signal bar: the last values known when the order goes in (the entry bar's close comes later)
    return trades.join(feature_labels().rename({'ts': 'signal_ts'}), on='signal_ts', how='left')


def sessions_of(trades: pl.DataFrame) -> pl.Series:
    """Every session from the trades' first to last date: a session without a trade is a 0 $ day in the statistics."""
    return read_training_data(start=trades['date'].min(), end=trades['date'].max(), columns=[])['date'].unique().sort()


def final_report(trades: pl.DataFrame, board: pl.DataFrame, sessions, final=FINAL, out_dir=None):
    """
    The final run: config.FINAL (strategy name, tp, sl), else the leaderboard's best by total net. Writes its
    statistics (by side, month, rolling month and label, csv), its charts and its strategy's TP x SL heatmaps, in
    out_dir (default results/final/).
    """
    out_dir = out_dir or os.path.join(RESULTS_DIR, 'final')
    strategy, tp, sl = final or board.select('strategy', 'tp', 'sl').row(0)
    run = trades.filter((pl.col('strategy') == strategy) & (pl.col('tp') == tp) & (pl.col('sl') == sl))
    if run.is_empty():
        raise ValueError(f"no trades for {(strategy, tp, sl)} in {TRADES_PATH}: check config.FINAL")
    stats = BacktestStats(run, sessions)
    os.makedirs(out_dir, exist_ok=True)
    stats.by_side().write_csv(os.path.join(out_dir, 'by_side.csv'))
    stats.by_period('month').write_csv(os.path.join(out_dir, 'by_month.csv'))
    stats.rolling().write_csv(os.path.join(out_dir, 'rolling_month.csv'))
    for col in LABELS:
        stats.by_label(col).write_csv(os.path.join(out_dir, f'by_{col}.csv'))
    BacktestPlots(stats, out_dir, title=f'{strategy}, TP {tp} SL {sl}').all()
    grid = board.filter(pl.col('strategy') == strategy)
    for value in ('total_net_usd', 'sharpe'):
        BacktestPlots.grid_heatmap(grid, value, out_dir, title=f'{strategy}: {value} by TP (y) and SL (x)')
    return (strategy, tp, sl), stats


def feature_labels(features=LABEL_FEATURES) -> pl.DataFrame:
    """
    Per 1-min bar (`ts`, joined on a trade's `signal_ts`): each training-data column in `features`, raw and as `{col}_q`, its
    quantile bucket over all bars (1 lowest .. n highest). The cut points use the whole history: fine to describe
    trades, not for a trading rule, which must take its cut points from the past only.
    """
    bars = pl.read_parquet(TRAINING_DATA_PATH, columns=['ts', *features])
    for col, n in features.items():
        cuts = [bars[col].quantile(i / n) for i in range(1, n)]
        bars = bars.with_columns(
            pl.col(col).cut(cuts, labels=[str(i) for i in range(1, n + 1)]).cast(pl.Utf8).cast(pl.Int64).alias(f'{col}_q'))
    return bars


def by_label(s: pl.DataFrame, cols) -> pl.DataFrame:
    """slices()' rows of one label (or label x side) with the label values as their own Int64 columns."""
    return (s.filter(pl.col('label') == ' x '.join(cols))
            .with_columns(pl.col('value').str.split(' / ').list.to_struct(fields=list(cols)))
            .unnest('value').with_columns(pl.col(list(cols)).cast(pl.Int64)).drop('label')
            .select('strategy', 'tp', 'sl', *cols, pl.exclude('strategy', 'tp', 'sl', *cols))
            .sort('strategy', 'tp', 'sl', *cols))


def _stats() -> list[pl.Expr]:
    net = pl.col('net_usd')
    return [
        pl.len().alias('trades'),
        pl.col('gross_usd').mean().alias('gross_per_trade'),
        net.mean().alias('net_per_trade'),
        (net > 0).mean().alias('win_rate'),
        net.sum().alias('net_usd'),
        (net.mean() / net.std() * pl.len().sqrt()).alias('t_net'),  # mean net per trade in standard errors
    ]


def slices(trades: pl.DataFrame, split=SPLIT) -> pl.DataFrame:
    """One row per strategy x tp x sl x slice (label, value): _stats() plus the halves and the years positive."""
    late = pl.col('date') >= datetime.date.fromisoformat(split)
    out = []
    for cols in SLICES:
        keys = ['strategy', 'tp', 'sl', *cols]
        halves = trades.group_by(keys).agg(
            *_stats(),
            trades_before=(~late).sum(), net_before=pl.col('net_usd').filter(~late).sum(),
            trades_after=late.sum(), net_after=pl.col('net_usd').filter(late).sum(),
        )
        years = (trades.group_by(*keys, 'year').agg(pl.col('net_usd').sum())
                 .group_by(keys).agg(years=pl.len(), years_positive=(pl.col('net_usd') > 0).sum()))
        out.append(halves.join(years, on=keys).select(
            'strategy', 'tp', 'sl',
            pl.lit(' x '.join(cols)).alias('label'),
            pl.concat_str([pl.col(c).cast(pl.Utf8) for c in cols], separator=' / ').alias('value'),
            *[c for c in halves.columns if c not in keys], 'years', 'years_positive',
        ))
    return pl.concat(out).sort('net_usd', descending=True)


def pooled(trades: pl.DataFrame) -> pl.DataFrame:
    """Per strategy x label value, all TP / SL pairs pooled: the label's net edge whatever the exits."""
    return pl.concat([
        trades.group_by('strategy', label).agg(*_stats()).select(
            'strategy', pl.lit(label).alias('label'), pl.col(label).alias('value'), pl.exclude('strategy', label))
        for label in LABELS
    ]).sort('strategy', 'label', 'value')


def chance_check(s: pl.DataFrame) -> dict:
    """
    Slices with MIN_TRADES in each half: how many are net positive in both, against how many chance would give if
    the halves were unrelated (share positive before x share positive after x slices); and of the slices positive
    before SPLIT, the share still positive after, against the share of all slices positive after.
    """
    j = s.filter((pl.col('trades_before') >= MIN_TRADES) & (pl.col('trades_after') >= MIN_TRADES))
    before, after = (j['net_before'] > 0), (j['net_after'] > 0)
    n = len(j)
    return {
        'slices': n,
        'positive_before': int(before.sum()),
        'positive_after': int(after.sum()),
        'positive_both': int((before & after).sum()),
        'expected_both_by_chance': before.mean() * after.mean() * n if n else float('nan'),
        'after_share_if_picked_before': float(after.filter(before).mean()) if before.any() else float('nan'),
        'after_share_all': float(after.mean()) if n else float('nan'),
    }


def _heatmap(ax, matrix, xlabels, ylabels, title):
    finite = matrix[np.isfinite(matrix)]
    cmap, norm = DIVERGING, None
    if len(finite) and finite.min() != finite.max():
        lo, hi = finite.min(), finite.max()
        if lo < 0 < hi:
            norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
        elif hi <= 0:
            cmap, norm = NEGATIVE, Normalize(vmin=lo, vmax=0)
        else:
            cmap, norm = POSITIVE, Normalize(vmin=0, vmax=hi)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(colors=TEXT_MUT, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, color=TEXT_SEC, fontsize=10, loc='left')
    # Label only the cells that clear costs: net > 0
    for i, j in zip(*np.nonzero(np.nan_to_num(matrix, nan=-1) > 0)):
        ax.text(j, i, _usd(matrix[i, j]), ha='center', va='center', color=TEXT_PRI, fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_MUT, length=0)
    cbar.ax.yaxis.set_major_formatter(USD)


def plot_hours(p: pl.DataFrame, names, out_dir):
    """hours.png: net $ per trade by hour (pooled over the pairs), one row per strategy, all trades / long / short."""
    hours = list(range(23))
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Net $ per trade by hour (shifted clock; New York = hour - 6), all TP / SL pairs pooled',
                 color=TEXT_PRI, fontsize=12, x=0.01, ha='left')
    for ax, (title, side) in zip(axes, [('All trades', None), ('Long', 1), ('Short', -1)]):
        m = np.full((len(names), len(hours)), np.nan)
        for i, name in enumerate(names):
            if side is None:
                r = p.filter((pl.col('strategy') == name) & (pl.col('label') == 'hour'))
            else:
                r = p.filter((pl.col('strategy') == name) & (pl.col('label') == 'hour x side') & (pl.col('side') == side))
            for h, v in zip(r['value'].to_list(), r['net_per_trade'].to_list()):
                m[i, h] = v
        _heatmap(ax, m, [str(h) for h in hours], names, title)
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, 'hours.png'))


def plot_labels(p: pl.DataFrame, names, out_dir):
    """labels.png: net $ per trade by session, regular hours, news, weekday and side (pairs pooled), per strategy."""
    panels = [('session', {1: 'Asia', 2: 'Europe', 3: 'US'}), ('rth', {0: 'outside RTH', 1: 'RTH'}),
              ('news_1', {0: 'no news', 1: 'news window'}), ('weekday', dict(enumerate(WEEKDAYS, 1))),
              ('side', {-1: 'short', 1: 'long'}),
              *[(f'{c}_q', {i: f'Q{i}' for i in range(1, n + 1)}) for c, n in LABEL_FEATURES.items()]]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.5), sharey=True)
    fig.suptitle('Net $ per trade by label, all TP / SL pairs pooled', color=TEXT_PRI, fontsize=12, x=0.01, ha='left')
    width = 0.8 / len(names)
    for ax, (label, names_of) in zip(axes, panels):
        values = sorted(names_of)
        for k, name in enumerate(names):
            r = p.filter((pl.col('strategy') == name) & (pl.col('label') == label))
            got = dict(zip(r['value'].to_list(), r['net_per_trade'].to_list()))
            ax.bar(np.arange(len(values)) + (k - (len(names) - 1) / 2) * width, [got.get(v, np.nan) for v in values],
                   width=width * 0.9, color=CATEGORICAL[k], label=name, zorder=3)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=2)
        ax.set_xticks(range(len(values)), [names_of[v] for v in values])
        ax.set_title(label, color=TEXT_SEC, fontsize=10, loc='left')
        _style(ax)
    # One legend row for all panels, under the title: never over a panel
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right', ncol=len(names), frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(TEXT_SEC)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, os.path.join(out_dir, 'labels.png'))


def plot_stability(s: pl.DataFrame, trades: pl.DataFrame, out_dir, top=6):
    """stability.png: net $ per year of the best slices that are positive in both halves (small multiples)."""
    best = s.filter((pl.col('trades_before') >= MIN_TRADES) & (pl.col('trades_after') >= MIN_TRADES)
                    & (pl.col('net_before') > 0) & (pl.col('net_after') > 0)).head(top)
    if best.is_empty():
        return None
    fig, axes = plt.subplots(1, len(best), figsize=(3 * len(best) + 1, 3.8), sharey=True, squeeze=False)
    fig.suptitle('Net $ per year: the best slices positive before and after the split', color=TEXT_PRI, fontsize=12,
                 x=0.01, ha='left')
    for ax, row in zip(axes[0], best.iter_rows(named=True)):
        cond = (pl.col('strategy') == row['strategy']) & (pl.col('tp') == row['tp']) & (pl.col('sl') == row['sl'])
        for col, v in zip(row['label'].split(' x '), row['value'].split(' / ')):
            cond &= pl.col(col) == int(v)
        y = trades.filter(cond).group_by('year').agg(pl.col('net_usd').sum()).sort('year')
        net = y['net_usd'].to_numpy()
        ax.bar(y['year'].to_list(), net, color=np.where(net >= 0, CATEGORICAL[0], CATEGORICAL[7]), width=0.8, zorder=3)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=2)
        ax.set_title(f"{row['strategy']} TP {row['tp']} SL {row['sl']}\n{row['label']} = {row['value']}",
                     color=TEXT_SEC, fontsize=9, loc='left')
        _style(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, 'stability.png'))


def main():
    trades = load_trades()
    names = trades['strategy'].unique(maintain_order=True).to_list()
    sessions = sessions_of(trades)
    out_dir = os.path.join(RESULTS_DIR, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    board = BacktestStats.grid(trades, by=('strategy', 'tp', 'sl'), sessions=sessions)
    board.write_csv(os.path.join(out_dir, 'leaderboard.csv'))
    s = slices(trades)
    p = pooled(trades)
    # pooled() keeps one `value` column; the label x side rows are rebuilt here for the hour chart
    p_side = pl.concat([
        trades.group_by('strategy', label, 'side').agg(*_stats()).select(
            'strategy', pl.lit(f'{label} x side').alias('label'), pl.col(label).alias('value'), 'side',
            pl.exclude('strategy', label, 'side'))
        for label in LABELS if label != 'side'
    ])
    both = pl.concat([p, p_side], how='diagonal')
    for cols in SLICES:
        by_label(s, cols).write_csv(os.path.join(out_dir, f"by_{'_x_'.join(cols)}.csv"))
    both.write_csv(os.path.join(out_dir, 'pooled.csv'))
    plot_hours(both, names, out_dir)
    plot_labels(p, names, out_dir)
    plot_stability(s, trades, out_dir)
    (strategy, tp, sl), stats = final_report(trades, board, sessions)

    c = chance_check(s)
    print(f"{len(trades):,} trades, {len(board)} runs ({TRADES_PATH}); {len(s):,} slices in {out_dir}/by_<label>.csv")
    with pl.Config(tbl_rows=10, tbl_cols=-1, float_precision=1, tbl_width_chars=220):
        print("\nLeaderboard by total net:")
        print(board.select('strategy', 'tp', 'sl', 'trades', 'win_rate', 'total_gross_usd', 'total_cost_usd',
                           'total_net_usd', 'max_drawdown_usd', 'sharpe').head(10))
    print(f"Judged (>= {MIN_TRADES} trades each side of {SPLIT}): {c['slices']:,}; net positive before {c['positive_before']:,}, "
          f"after {c['positive_after']:,}, both {c['positive_both']:,} (chance alone: {c['expected_both_by_chance']:.0f})")
    print(f"Positive after {SPLIT}: {c['after_share_if_picked_before']:.1%} of the slices positive before it, "
          f"{c['after_share_all']:.1%} of all slices")
    with pl.Config(tbl_rows=15, tbl_cols=-1, float_precision=1, tbl_width_chars=220):
        good = s.filter((pl.col('trades_before') >= MIN_TRADES) & (pl.col('trades_after') >= MIN_TRADES)
                        & (pl.col('net_before') > 0) & (pl.col('net_after') > 0))
        print(good.select('strategy', 'tp', 'sl', 'label', 'value', 'trades', 'net_per_trade', 'win_rate', 't_net',
                          'net_before', 'net_after', 'years_positive', 'years').head(15))
        print(f"\nFinal run {strategy}, TP {tp} SL {sl} ({RESULTS_DIR}/final/):")
        print(stats.by_side().select('side', 'trades', 'win_rate', 'total_gross_usd', 'total_net_usd', 'profit_factor',
                                     'max_drawdown_usd', 'sharpe'))


if __name__ == "__main__":
    main()
