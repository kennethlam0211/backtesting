"""
Walk-forward selection: the long side and the short side each pick, independently, their best run (signal x stops,
1 contract) over the previous `train` periods, by net PnL or Calmar; both picks trade the next `test` periods, then it
rolls on by `test`. Periods are trading hours, days or months. Only the traded (test) periods are recorded; a training
period never counts.

    python -m backtesting.walk_forward                                        # train 5 days, trade the next 5, by PnL
    python -m backtesting.walk_forward --train 5 --test 1                     # retrain every day
    python -m backtesting.walk_forward --unit month --train 1,3,6,12 --metric calmar
    python -m backtesting.walk_forward --train 1 --test 1 --opposite          # trade the opposite of each pick
    python -m backtesting.walk_forward --unit hour --train 5,8 --test 1       # the last 5 / 8 trading hours

Input: a sweep's trades (every run with `signal`, `freq`, `stops`, `side`, `entry_ts`, `net_usd`; `exit_ts` for hours),
by default results/sweep/limit/trades_*.parquet. Every run is a candidate each time; nothing is filtered on the whole
period, which would see the future. Each time, from the training window only, per side: the runs net positive with at
least `min_trades` trades (default max(2, train) for months, else 1), and of those the best by the metric: pnl (net)
or calmar (net / max drawdown of the window's net per session, or per hour; a window with no drawdown ranks above every
other, by net). A trade counts in training once its PnL is known: for days and months its session (every trade closes
by the session's end); for hours the hour its exit bar closes (exit_ts, the exit bar's start, + the bar), so a trade
still open when the test hour starts is not seen. A test period trades the picks' signals that enter in it; with
hours such a trade can still be open in the next hour, next to the next picks'.
--opposite trades each pick's mirror instead: the other side with TP and SL swapped (its `_reversed` signal and stops
'sl/tp' in the sweep). Its levels are the pick's (the pick's TP is the mirror's SL and the other way round), so the
gross PnL is the pick's with the sign flipped; only the commission and the other side's limit fill differ.
Writes to --out (default results/walk_forward/): the picks, the traded trades and acc_pnl_*.png; prints the statistics
of both sides together and of each side (the side traded).
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from params import RESULTS_DIR

TEST_FROM = np.datetime64('2026-01-01')  # shown apart in the statistics and the plot
HOURS = 23  # a session is [00:00, 23:00) on the shifted clock


def load(pattern: str):
    """The sweep's trades (with `k`, the run, and `date`, the session), the runs and the sessions."""
    t = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(pattern))]).with_columns(date=pl.col('entry_ts').dt.date())
    runs = t.select('signal', 'stops', 'side').unique().sort('signal', 'stops').with_row_index('k')
    t = t.join(runs.select('signal', 'stops', 'k'), on=['signal', 'stops'])
    return t, runs, t['date'].unique().sort()


def slots(t: pl.DataFrame, dates: pl.Series, unit: str):
    """
    Each trade's slot at entry and the slot from whose end its PnL is known, and the number of slots. A slot is a session
    (days, months) or an hour of one (hours: session row x HOURS + the shifted clock's hour).
    """
    session = dates.search_sorted(t['date']).to_numpy()
    if unit != 'hour':
        return session, session, len(dates)
    if 'exit_ts' not in t.columns:
        raise ValueError("hour windows need the trades' exit_ts: rerun the sweep with it")
    # Known once the exit bar has closed: exit_ts is the bar's start
    known = t.select(pl.col('exit_ts') + pl.duration(minutes=pl.col('freq').cast(pl.Int64)) - pl.duration(seconds=1)).to_series()
    entry = session * HOURS + t['entry_ts'].dt.hour().to_numpy()
    return entry, dates.search_sorted(known.dt.date()).to_numpy() * HOURS + known.dt.hour().to_numpy(), len(dates) * HOURS


def matrices(t: pl.DataFrame, known: np.ndarray, n_slots: int, n_runs: int):
    """Every run's net and trade count per slot, by the slot its PnL is known in (slots x runs)."""
    g = t.select(pl.Series('slot', known), 'k', 'net_usd').group_by('slot', 'k').agg(pl.col('net_usd').sum(), pl.len().alias('n'))
    # float64: float32 rounding flips near-ties between runs' nets, so the picks would depend on the precision
    net, count = np.zeros((n_slots, n_runs)), np.zeros((n_slots, n_runs), np.int32)
    net[g['slot'].to_numpy(), g['k'].to_numpy()] = g['net_usd'].to_numpy()
    count[g['slot'].to_numpy(), g['k'].to_numpy()] = g['n'].to_numpy()
    return net, count


def score(net, metric):
    """Every run's score on the window's net per slot: pnl = its net; calmar = net / max drawdown (1e9 + net with none)."""
    cum = np.cumsum(net, axis=0)
    if metric == 'pnl':
        return cum[-1]
    mdd = (np.maximum.accumulate(np.maximum(cum, 0.0), axis=0) - cum).max(axis=0)
    return np.where(mdd > 0, cum[-1] / np.where(mdd > 0, mdd, 1), 1e9 + cum[-1])


def pick(net, count, side, min_trades, metric):
    """The best long run and the best short run on these rows (the training window), each on its own; None if none."""
    ok = (net.sum(axis=0) > 0) & (count.sum(axis=0) >= min_trades)
    s = score(net, metric)
    best = []
    for sd in (1, -1):
        cand = np.flatnonzero(ok & (side == sd))
        best.append(int(cand[np.argmax(s[cand])]) if len(cand) else None)
    return tuple(best)


def mirrors(runs: pl.DataFrame) -> list:
    """Each run's mirror: the same condition on the other side (`_reversed` toggled), TP and SL swapped; None if absent."""
    key = {(g, st): k for k, g, st in zip(runs['k'], runs['signal'], runs['stops'])}
    out = []
    for g, st in zip(runs['signal'], runs['stops']):
        tp, sl = st.split('/')
        other = g.removesuffix('_reversed') if g.endswith('_reversed') else g + '_reversed'
        out.append(key.get((other, f'{sl}/{tp}')))
    return out


def walk(net, count, side, period, train, test, min_trades, metric):
    """
    From the first full training window, `test` periods at a time. period: each slot's period number (0, 1, ...,
    ascending). Returns [(block, first slot, (long, short))] and each slot's test block (-1 before the first).
    """
    picks, block = [], np.full(len(period), -1)
    for b, i in enumerate(range(train, period.max() + 1, test)):
        lo, mid, hi = np.searchsorted(period, [i - train, i, i + test])  # training rows lo:mid, test rows mid:hi
        block[mid:hi] = b
        picks.append((b, int(mid), pick(net[lo:mid], count[lo:mid], side, min_trades, metric)))
    return picks, block


def _daily(trades: pl.DataFrame, sessions: pl.Series) -> np.ndarray:
    """The trades' net per session, 0 on sessions without one."""
    return (pl.DataFrame({'date': sessions}).join(trades.group_by('date').agg(pl.col('net_usd').sum()), on='date', how='left')
            .fill_null(0).sort('date')['net_usd'].to_numpy())


def statistics(trades: pl.DataFrame, sessions: pl.Series) -> dict:
    """The traded periods' statistics: trades, win rate, PnL, drawdown, Calmar, Sharpe, positive days / months."""
    n = trades['net_usd']
    x = _daily(trades, sessions)
    cum = np.cumsum(x)
    mdd = float((np.maximum.accumulate(np.maximum(cum, 0.0)) - cum).max()) if len(x) else 0.0
    years = (sessions.max() - sessions.min()).days / 365.25
    by_day = pl.DataFrame({'date': sessions.sort(), 'net': x})
    months = by_day.group_by(pl.col('date').dt.truncate('1mo')).agg(pl.col('net').sum())['net']
    traded_days = by_day.filter(pl.col('date').is_in(trades['date'].unique().implode()))['net']
    wins, losses = n.filter(n > 0), n.filter(n <= 0)
    test = trades.filter(pl.col('entry_ts') >= TEST_FROM)['net_usd']
    return {
        'trades': len(trades), 'win_rate': round(float((n > 0).mean()), 3) if len(n) else None,
        'avg_win': round(float(wins.mean()), 1) if len(wins) else None,
        'avg_loss': round(float(losses.mean()), 1) if len(losses) else None,
        'profit_factor': round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        'net': round(float(n.sum())), 'net_per_year': round(float(n.sum() / years)) if years > 0 else None,
        'max_dd': round(mdd), 'calmar': round(float(n.sum() / years / mdd), 2) if mdd > 0 and years > 0 else None,
        'sharpe': round(float(x.mean() / x.std(ddof=1) * np.sqrt(252)), 2) if len(x) > 1 and x.std() > 0 else None,
        'positive_days': round(float((traded_days > 0).mean()), 3) if len(traded_days) else None,
        'positive_months': round(float((months > 0).mean()), 3) if len(months) else None,
        'net_2026': round(float(test.sum())), 'win_rate_2026': round(float((test > 0).mean()), 3) if len(test) else None,
    }


def plot(results, out_path):
    """One panel per setup: both sides' accumulated net (drawdown shaded, 2026 in another colour) and each side's."""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.4 * len(results)), squeeze=False)
    for ax, (label, trades, sessions, s) in zip(axes[:, 0], results):
        x = sessions.to_numpy()
        cum = np.cumsum(_daily(trades, sessions))
        new = x >= TEST_FROM
        k = int(new.argmax()) if new.any() else len(x)
        ax.fill_between(x, cum, np.maximum.accumulate(np.maximum(cum, 0.0)), color='#d62728', alpha=0.15, lw=0)
        for sd, color, name in ((1, '#2ca02c', 'long side'), (-1, '#9467bd', 'short side')):
            ax.plot(x, np.cumsum(_daily(trades.filter(pl.col('side') == sd), sessions)), color=color, lw=0.9, alpha=0.8,
                    label=name)
        ax.plot(x[:k + 1], cum[:k + 1], color='#2a6fdb', lw=1.9, label='both, to 2025')
        ax.plot(x[k:], cum[k:], color='#e8762b', lw=1.9, label='both, 2026')
        ax.axhline(0, color='#999', lw=0.8)
        ax.set_title(f"{label}: {s['trades']} trades, win {s['win_rate']:.0%}, net \\${s['net']:,} "
                     f"(\\${s['net_per_year']:,}/year), max DD \\${s['max_dd']:,}, Calmar {s['calmar']}, Sharpe {s['sharpe']}, "
                     f"positive months {s['positive_months']:.0%}; 2026 \\${s['net_2026']:,}", fontsize=9, loc='left')
        ax.set_ylabel('acc net $')
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(loc='upper left', frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--trades', default=os.path.join(RESULTS_DIR, 'sweep', 'limit', 'trades_*.parquet'),
                        help="the sweep's trades (glob)")
    parser.add_argument('--unit', choices=('hour', 'day', 'month'), default='day',
                        help='period: trading hour, trading day or calendar month')
    parser.add_argument('--train', default='5', help='training periods, comma-separated (e.g. 5 or 1,3,6,12)')
    parser.add_argument('--test', type=int, default=5, help='periods traded before the next pick')
    parser.add_argument('--metric', choices=('pnl', 'calmar'), default='pnl', help='how each side picks its run')
    parser.add_argument('--min-trades', type=int, default=None, help='per candidate run in the training window')
    parser.add_argument('--opposite', action='store_true', help="trade each pick's mirror: other side, TP / SL swapped")
    parser.add_argument('--out', default=os.path.join(RESULTS_DIR, 'walk_forward'))
    args = parser.parse_args(argv)

    t, runs, dates = load(args.trades)
    entry_slot, known_slot, n_slots = slots(t, dates, args.unit)
    net, count = matrices(t, known_slot, n_slots, len(runs))
    side = runs['side'].to_numpy()
    name = [f"{'buy' if s == 1 else 'sell'} {g} {st}" for s, g, st in zip(side, runs['signal'], runs['stops'])]
    if args.unit == 'month':
        period = np.unique(dates.dt.truncate('1mo').to_numpy(), return_inverse=True)[1]
    else:
        period = np.arange(n_slots)
    slot_name = (lambda i: f'{dates[i // HOURS]} {i % HOURS:02d}:00') if args.unit == 'hour' else (lambda i: str(dates[i]))
    mirror = mirrors(runs)
    opp = '_opposite' if args.opposite else ''
    os.makedirs(args.out, exist_ok=True)

    rows, plots = [], []
    for train in (int(x) for x in args.train.split(',')):
        min_trades = args.min_trades if args.min_trades is not None else (max(2, train) if args.unit == 'month' else 1)
        picked, block = walk(net, count, side, period, train, args.test, min_trades, args.metric)
        # The runs traded: the picks, or their mirrors
        picks = [(b, i, tuple(None if k is None else mirror[k] for k in p)) for b, i, p in picked] if args.opposite else picked
        legs = [(b, k) for b, _, p in picks for k in p if k is not None]  # each test block's traded runs
        chosen = pl.DataFrame({'block': np.array([b for b, _ in legs], dtype=np.int64),
                               'k': np.array([k for _, k in legs], dtype=np.uint32)})
        trades = t.with_columns(block=pl.Series(block[entry_slot], dtype=pl.Int64)).join(chosen, on=['block', 'k'])
        tested = block.reshape(len(dates), HOURS).max(axis=1) if args.unit == 'hour' else block
        sessions = dates.filter(pl.Series(tested >= 0))  # the test periods' sessions only
        label = (f"{args.metric}, train {train} {args.unit}{'s' if train > 1 else ''}, trade {args.test}"
                 + (', opposite' if args.opposite else ''))
        for part, sd in (('both', 0), ('long', 1), ('short', -1)):
            sel = trades if sd == 0 else trades.filter(pl.col('side') == sd)
            runs_used = {k for *_, p in picks for k in p if k is not None and sd in (0, side[k])}
            rows.append({'setup': label, 'side': part, 'from': str(sessions.min()), 'tests': len(picks),
                         'distinct_runs': len(runs_used), **statistics(sel, sessions)})
        plots.append((label, trades, sessions, rows[-3]))
        tag = f"{args.metric}_{train}{args.unit[0]}_{args.test}{args.unit[0]}{opp}"
        label_of = lambda k: None if k is None else name[k]
        pl.DataFrame([(slot_name(i), label_of(p[0]), label_of(p[1]), label_of(q[0]), label_of(q[1]))
                      for (_, i, p), (*_, q) in zip(picked, picks)],
                     schema=['from', 'long_pick', 'short_pick', 'traded_for_long_pick', 'traded_for_short_pick'],
                     orient='row').write_csv(os.path.join(args.out, f'picks_{tag}.csv'))
        trades.drop('k', 'block').write_parquet(os.path.join(args.out, f'trades_{tag}.parquet'))

    u = args.unit[0]
    out_png = os.path.join(args.out, f"acc_pnl_{args.metric}_{args.train.replace(',', '-')}{u}_{args.test}{u}{opp}.png")
    plot(plots, out_png)
    with pl.Config(tbl_rows=30, tbl_cols=-1, tbl_width_chars=260, tbl_hide_dataframe_shape=True, fmt_str_lengths=40):
        stats = pl.DataFrame(rows)
        print('Traded (test) periods only, 1 contract per side:')
        print(stats.select('setup', 'side', 'from', 'tests', 'distinct_runs', 'trades', 'win_rate', 'avg_win', 'avg_loss',
                           'profit_factor'))
        print(stats.select('setup', 'side', 'net', 'net_per_year', 'max_dd', 'calmar', 'sharpe', 'positive_days',
                           'positive_months', 'net_2026', 'win_rate_2026'))
    print(f'picks, trades and {out_png} in {args.out}/')


if __name__ == '__main__':
    main()
