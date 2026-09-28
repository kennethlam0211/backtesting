"""
Walk-forward test of ranking measures: which weighting of Sharpe, win rate and Martin picks runs that make money in the
most test months?

    python -m backtesting.metric_search ud_range                     # results/sweep/ud_range/
    python -m backtesting.metric_search limit --top 3 --step 0.25 --train 12
    python -m backtesting.metric_search ud_range ud_near             # several sweeps' runs together
    python -m backtesting.metric_search ud_range --weights sharpe=1,martin=1,months=1 --smooth --train 24 --top 1
    python -m backtesting.metric_search ud_range --unit day --train 3 --validate --min-trades 1   # in sessions
    python -m backtesting.metric_search ud_range --unit day --train 252 --anchored --repick-loss 2000

Every month from the --train'th on: train on the --train months before it, then trade it (the test month); only test
months are recorded. In training, per run (signal x stops, at least --min-trades trades in the window): Sharpe (daily
net over every session, x sqrt(252)), Sortino (the same over the downside deviation: only losing days count as risk),
win rate (per trade), Martin (net per year / Ulcer index), Calmar and months (the share of the window's months it made
money in); --metrics picks the ones mixed (default sharpe, win, martin). Each measure becomes its percentile rank among
the side's candidates, so their scales do not matter; a run's score is the weighted sum of its ranks (weights on a grid
of --step, summing to 1, or the one --weights; plus pure Calmar as a baseline). --smooth: each rank is first averaged
over the run and its neighbours in the stop grid (same signal and grid, target and / or stop one step away), so a run
ranks high only on a plateau of good stops, not as one lucky setting. --validate: the month before the test month is
held out of training as a validation month (--validate N: N months), and only runs that made money over it (net > 0)
can be picked. --every K:
new picks every K months, held in between; --anchored: trained on all months so far (at least --train).
--keep-winners: at each --every check, a weighting keeps picks that made money since the last check (retrains the
others), e.g. --train 36 --every 6 --keep-winners: 3 years of training, checked every half year.
--repick-loss X: instead of --every, each weighting keeps its picks until they have lost $X since they were made,
then picks again (with --anchored: the first pick on --train periods, later ones on all history so far).
--unit day: every period above is a session instead of a month (train 3 days, validate 1, test 1, ...); the test days
are still added up per calendar month for the consistency figures; 'months' is then the share of winning days. Per condition
and side the best-scoring run; the top --top of each side trade the test month, 1 contract each.
Consistency per weighting: the share of positive test months, the mean and worst month, the total, and the max
drawdown over the test sessions. Weightings are ranked on the test months before TEST_FROM (2026: positive share, then
mean month), so the 2026 months then test that choice.
Writes metric_search{_tag}.parquet (per weighting), metric_search_months{_tag}.parquet (per weighting and test month:
net and picks) and acc_pnl_metric_search{_tag}.png to the first sweep folder.
"""
import argparse
import glob
import itertools
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import sparse

from backtesting.pairs import TEST_FROM, calmar, martin
from params import RESULTS_DIR

MEASURES = ('sharpe', 'sortino', 'win', 'martin', 'calmar', 'months')


def load(sweeps: list[str]) -> tuple[str, pl.DataFrame]:
    """The sweeps' trades (the columns needed here), with their session date; and the first sweep's folder."""
    folders = [f if os.path.isdir(f) else os.path.join(RESULTS_DIR, 'sweep', f) for f in sweeps]
    files = [f for d in folders for f in sorted(glob.glob(os.path.join(d, 'trades_*.parquet')))]
    t = pl.concat([pl.read_parquet(f, columns=['signal', 'stops', 'side', 'entry_ts', 'net_usd']) for f in files])
    return folders[0], t.with_columns(date=pl.col('entry_ts').dt.date())


def neighbours(runs: pl.DataFrame) -> sparse.csr_matrix:
    """
    runs x runs, 1 where two runs share the signal and the stop grid (fixed 'tp/sl' units or vol 'tpx/slx') and their
    target and stop are each at most one step apart on that grid (a run is its own neighbour).
    """
    grid = runs.with_columns(vol=pl.col('stops').str.contains('x'),
                             tp=pl.col('stops').str.extract(r'^([\d.]+)').cast(pl.Float64),
                             sl=pl.col('stops').str.extract(r'/([\d.]+)').cast(pl.Float64))
    grid = grid.with_columns(ti=pl.col('tp').rank('dense').over('signal', 'vol').cast(pl.Int32),  # signed: steps
                             si=pl.col('sl').rank('dense').over('signal', 'vol').cast(pl.Int32))  # can be negative
    pairs = (grid.join(grid, on=['signal', 'vol'], suffix='_n')
             .filter(((pl.col('ti') - pl.col('ti_n')).abs() <= 1) & ((pl.col('si') - pl.col('si_n')).abs() <= 1)))
    k, kn = pairs['k'].to_numpy(), pairs['k_n'].to_numpy()
    return sparse.csr_matrix((np.ones(len(k)), (k, kn)), shape=(len(runs), len(runs)))


class Book:
    """Every run (signal x stops) as column k: daily net (sessions x runs); trades, wins and net per period (periods x
    runs), a period being a month or, with unit 'day', a session."""

    def __init__(self, t: pl.DataFrame, unit: str = 'month'):
        runs = t.select('signal', 'stops', 'side').unique().sort('signal', 'stops').with_row_index('k')
        self.names = [f"{'buy' if sd == 1 else 'sell'} {s} {st}"
                      for s, st, sd in runs.select('signal', 'stops', 'side').iter_rows()]
        self.side = runs['side'].to_numpy()
        _, self.cond = np.unique(runs['signal'].str.replace(r'_(long|short|reversed)$', '').to_numpy(), return_inverse=True)
        self.dates = t['date'].unique().sort()
        self.per_year = 12 if unit == 'month' else 252
        key = self.dates.dt.truncate('1mo') if unit == 'month' else self.dates
        self.periods = key.unique().sort()
        self.period_of = self.periods.search_sorted(key).to_numpy()  # per session
        t = t.join(runs.select('signal', 'stops', 'k'), on=['signal', 'stops'])
        g = t.group_by('k', 'date').agg(pl.col('net_usd').sum())
        self.D = np.zeros((len(self.dates), len(runs)))
        self.D[self.dates.search_sorted(g['date']).to_numpy(), g['k'].to_numpy()] = g['net_usd'].to_numpy()
        period = pl.col('date').dt.truncate('1mo') if unit == 'month' else pl.col('date')
        m = t.group_by('k', period.alias('period')).agg(pl.len().alias('n'), (pl.col('net_usd') > 0).sum().alias('w'))
        at = (self.periods.search_sorted(m['period']).to_numpy(), m['k'].to_numpy())
        self.N, self.W = np.zeros((len(self.periods), len(runs))), np.zeros((len(self.periods), len(runs)))
        self.N[at], self.W[at] = m['n'].to_numpy(), m['w'].to_numpy()
        self.M = np.zeros((len(self.periods), len(runs)))  # net per period
        np.add.at(self.M, self.period_of, self.D)
        self.near = neighbours(runs)

    def measures(self, first: int, last: int, min_trades: int):
        """Periods [first, last): each run's MEASURES, and whether it has min_trades trades."""
        x = self.D[(self.period_of >= first) & (self.period_of < last)]
        yrs = (last - first) / self.per_year
        n, w = self.N[first:last].sum(axis=0), self.W[first:last].sum(axis=0)
        sd, down = x.std(axis=0, ddof=1), np.sqrt((np.minimum(x, 0) ** 2).mean(axis=0))
        sharpe = np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(252), np.nan)
        sortino = np.where(down > 0, x.mean(axis=0) / np.where(down > 0, down, 1) * np.sqrt(252), np.nan)
        win = np.where(n > 0, w / np.where(n > 0, n, 1), np.nan)
        months = (self.M[first:last] > 0).mean(axis=0)
        return ({'sharpe': sharpe, 'sortino': sortino, 'win': win, 'martin': martin(x, yrs)[0], 'calmar': calmar(x, yrs)[0],
                 'months': months}, n >= min_trades)


def pct_rank(v: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1], ties averaged; NaN ranks lowest."""
    return pl.Series(np.where(np.isnan(v), -np.inf, v)).rank('average').to_numpy() / len(v)


def side_ranks(book: Book, values: dict, idx: np.ndarray, smooth: bool) -> dict:
    """Each measure's percentile rank among the candidates `idx`; with `smooth`, averaged over each run's candidate
    neighbours in the stop grid."""
    out = {}
    for m, v in values.items():
        r = pct_rank(v[idx])
        if smooth:
            full, has = np.zeros(book.near.shape[0]), np.zeros(book.near.shape[0])
            full[idx], has[idx] = r, 1.0
            r = (book.near @ full)[idx] / (book.near @ has)[idx]
        out[m] = r
    return out


def pick(book: Book, ranks: dict, idx: np.ndarray, weights: dict, top: int) -> np.ndarray:
    """Candidates `idx` of one side: the best-scoring run per condition, the top `top` of those (best first)."""
    score = sum(w * ranks[m] for m, w in weights.items() if w)
    order = idx[np.argsort(-score, kind='stable')]
    _, first = np.unique(book.cond[order], return_index=True)
    return order[np.sort(first)][:top]


def label(weights: dict) -> str:
    return ' '.join(f'{m} {w:g}' for m, w in weights.items() if w)


def walk(book: Book, grid: list[dict], train: int, top: int, min_trades: int, smooth: bool = False,
         validate: int = 0, every: int = 1, anchored: bool = False, repick_loss: float | None = None,
         keep_winners: bool = False):
    """Every weighting's test periods: per period its net (long / short) and picks, and its daily net on test sessions.
    The picks are made every `every` periods and held in between; `anchored`: trained on all periods so far.
    repick_loss: instead of every `every` periods, a weighting picks again once its picks have lost that much ($) since
    they were made (checked at each period's end). keep_winners: at each `every` check, a weighting whose picks made
    money since the last check keeps them; the others pick again."""
    rows, daily = [], {label(w): np.zeros(len(book.dates)) for w in grid}
    held, since, lately = {}, {}, {}  # lately: net since the last `every` check
    first = train + validate
    for i in range(first, len(book.periods)):
        if repick_loss is None:
            check = (i - first) % every == 0
            due = [label(w) for w in grid if check and (not keep_winners or lately.get(label(w), 0.0) <= 0)]
            if check:
                lately = {label(w): 0.0 for w in grid}
        else:
            due = [label(w) for w in grid if label(w) not in held or since[label(w)] <= -repick_loss]
        if due:
            end = i - validate  # training ends before the validation periods, if any
            values, ok = book.measures(0 if anchored else end - train, end, min_trades)
            if validate:
                ok = ok & (book.M[end:i].sum(axis=0) > 0)  # made money over the held-out periods
            sides = {sd: np.flatnonzero(ok & (book.side == sd)) for sd in (1, -1)}
            ranks = {sd: side_ranks(book, values, idx, smooth) for sd, idx in sides.items() if len(idx)}
            for w in grid:
                if label(w) in due:  # no candidate on either side: no trades
                    held[label(w)] = np.concatenate([np.zeros(0, dtype=np.int64)] + [
                        pick(book, ranks[sd], idx, w, top) for sd, idx in sides.items() if len(idx)])
                    since[label(w)] = 0.0
        sess = book.period_of == i
        for weights in grid:
            chosen = held[label(weights)]
            x = book.D[sess][:, chosen]
            daily[label(weights)][sess] = x.sum(axis=1)
            net, sd = x.sum(axis=0), book.side[chosen]
            since[label(weights)] += float(net.sum())
            lately[label(weights)] = lately.get(label(weights), 0.0) + float(net.sum())
            p = book.periods[i]
            rows.append({'weights': label(weights), 'period': p, 'month': p.replace(day=1), 'net': float(net.sum()),
                         'repick': label(weights) in due,
                         'long_net': float(net[sd == 1].sum()), 'short_net': float(net[sd == -1].sum()),
                         'picks': ', '.join(book.names[k] for k in chosen)})
    return pl.DataFrame(rows), daily


def to_months(periods: pl.DataFrame) -> pl.DataFrame:
    """The test periods added up per weighting and calendar month."""
    return (periods.group_by('weights', 'month', maintain_order=True)
            .agg(pl.col('net', 'long_net', 'short_net', 'repick').sum()))


def summary(months: pl.DataFrame, daily: dict, book: Book) -> pl.DataFrame:
    """Per weighting, test months before TEST_FROM and from it: positive share, mean / worst month, net, max drawdown."""
    test = (book.dates >= TEST_FROM).to_numpy()
    tested = (book.dates >= months['month'].min()).to_numpy()
    out = []
    for w, g in months.group_by('weights', maintain_order=True):
        row = {'weights': w[0]}
        for tag, part, sess in (('to_2025', g.filter(pl.col('month') < TEST_FROM), tested & ~test),
                                ('2026', g.filter(pl.col('month') >= TEST_FROM), test)):
            if part.is_empty():  # e.g. picked once just before TEST_FROM: no earlier test month
                row |= {f'positive_{tag}': None, f'months_{tag}': 0, f'mean_{tag}': None, f'worst_{tag}': None,
                        f'net_{tag}': 0.0, f'max_dd_{tag}': None, f'repicks_{tag}': 0}
                continue
            _, (mdd,) = calmar(daily[w[0]][sess][:, None], 1)
            row |= {f'positive_{tag}': float((part['net'] > 0).mean()), f'months_{tag}': part.height,
                    f'mean_{tag}': float(part['net'].mean()), f'worst_{tag}': float(part['net'].min()),
                    f'net_{tag}': float(part['net'].sum()), f'max_dd_{tag}': float(mdd),
                    f'repicks_{tag}': int(part['repick'].sum())}
        out.append(row)
    return (pl.DataFrame(out).with_columns(pl.col(pl.Null).cast(pl.Float64))  # all empty: still numbers
            .sort('positive_to_2025', 'mean_to_2025', 'net_2026', descending=True, nulls_last=True))


def plot(res: pl.DataFrame, months: pl.DataFrame, daily: dict, book: Book, out_path: str, title: str):
    """Top: accumulated test net of the two most consistent weightings, pure Martin and pure Calmar. Bottom: the most
    consistent one's test months."""
    x = book.dates.to_numpy()
    start = book.dates.search_sorted(months['month'].min())
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={'height_ratios': [3, 2]})
    pct = lambda v: 'n/a' if v is None else f'{v:.0%}'  # no test month in that part
    have = set(res['weights'])
    shown = [w for w in dict.fromkeys([*res['weights'].head(2), 'martin 1', 'calmar 1']) if w in have]
    for w in shown:
        r = res.row(res['weights'].to_list().index(w), named=True)
        ax.plot(x[start:], np.cumsum(daily[w][start:]), lw=2 if w == shown[0] else 1.1,
                label=f"{w}: {pct(r['positive_to_2025'])} months up to 2025, {pct(r['positive_2026'])} in 2026; "
                      f"2026 \\${r['net_2026']:,.0f}")
    ax.axvline(np.datetime64(TEST_FROM), color='#999', lw=0.8, ls='--')
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_title(title, fontsize=10, loc='left')
    ax.set_ylabel('acc test net $')
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    g = months.filter(pl.col('weights') == shown[0])
    bx.bar(g['month'].to_numpy(), g['net'].to_numpy(), width=25,
           color=np.where(g['net'].to_numpy() > 0, '#2ca02c', '#d62728'))
    bx.axhline(0, color='#999', lw=0.8)
    bx.set_title(f'{shown[0]}: net per test month', fontsize=10, loc='left')
    bx.set_ylabel('month net $')
    for a in (ax, bx):
        a.grid(alpha=0.3)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('sweep', nargs='+', help='sweep folders: names under results/sweep/ (e.g. ud_range) or paths')
    parser.add_argument('--unit', choices=('month', 'day'), default='month',
                        help='the period of --train, --every, the validation and the test (day: one session)')
    parser.add_argument('--train', type=int, default=12, help='training periods before each test period')
    parser.add_argument('--top', type=int, default=3, help='picks per side each month')
    parser.add_argument('--step', type=float, default=0.25, help='weight grid step')
    parser.add_argument('--min-trades', type=int, default=20, help='per run in the training window')
    parser.add_argument('--metrics', default='sharpe,win,martin', help=f'the measures mixed, from {",".join(MEASURES)}')
    parser.add_argument('--weights', default=None, help='one weighting instead of the grid, e.g. sharpe=1,martin=1 '
                                                        '(normalised to sum 1)')
    parser.add_argument('--smooth', action='store_true', help='rank on plateaus: average over neighbouring stops')
    parser.add_argument('--every', type=int, default=1, help='pick every this many periods, hold the picks between')
    parser.add_argument('--anchored', action='store_true', help='train on all periods so far, not a rolling window')
    parser.add_argument('--keep-winners', action='store_true',
                        help='at each --every check, keep the picks if they made money since the last one')
    parser.add_argument('--repick-loss', type=float, default=None,
                        help='pick again once the picks have lost this many $ since they were made (not --every)')
    parser.add_argument('--validate', type=int, nargs='?', const=1, default=0,
                        help='hold out this many periods (bare: 1) before each test period; pick only runs that made '
                             'money over them')
    parser.add_argument('--tag', default=None, help='a suffix for the output file names')
    args = parser.parse_args(argv)

    folder, t = load(args.sweep)
    book = Book(t, args.unit)
    ws = np.round(np.arange(0, 1 + 1e-9, args.step), 6)
    if args.weights:
        given = {m: float(w) for m, w in (kv.split('=') for kv in args.weights.split(','))}
        metrics, grid = list(given), [{m: w / sum(given.values()) for m, w in given.items()}]
    else:
        metrics = args.metrics.split(',')
        grid = [dict(zip(metrics, w)) for w in itertools.product(ws, repeat=len(metrics)) if abs(sum(w) - 1) < 1e-9]
    assert set(metrics) <= set(MEASURES), f'measures: from {MEASURES}'
    periods, daily = walk(book, grid + [{'calmar': 1.0}] * ('calmar' not in metrics), args.train, args.top,
                          args.min_trades, args.smooth, args.validate, args.every, args.anchored, args.repick_loss,
                          args.keep_winners)
    months = to_months(periods)
    res = summary(months, daily, book)
    tag = f'_{args.tag}' if args.tag else ''
    res.write_parquet(os.path.join(folder, f'metric_search{tag}.parquet'))
    periods.write_parquet(os.path.join(folder, f'metric_search_months{tag}.parquet'))
    out = os.path.join(folder, f'acc_pnl_metric_search{tag}.png')
    u = args.unit
    plot(res, months, daily, book, out, f"{' + '.join(args.sweep)}: train {args.train} {u}s"
                                        f"{f', validate {args.validate} {u}' if args.validate else ''}, trade the next {u}, top "
                                        f"{args.top} buys + top {args.top} sells (only test {u}s)")

    pl.Config.set_tbl_rows(80).set_tbl_width_chars(250).set_tbl_hide_dataframe_shape(True).set_fmt_str_lengths(60)
    pl.Config.set_tbl_cols(20)
    print(f"{args.sweep}: train {args.train} {u}s, test the next; test months {months['month'].min()} .. "
          f"{months['month'].max()}. Most consistent before {TEST_FROM} first:")
    print(res.with_columns(pl.col('^positive_.*$').round(2), pl.col('^(mean|worst|net|max_dd)_.*$').round(0)))
    print(out)


if __name__ == '__main__':
    main()
