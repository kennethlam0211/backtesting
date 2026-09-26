"""
Statistics of a backtest's trades (backtesting/backtest.py's output after add_costs), in $:

    stats = BacktestStats(trades, sessions=df['date'].unique())
    stats.summary()               # one dict: counts, win rate, gross / net PnL (all, long, short), drawdown, Sharpe, ...
    stats.by_side()               # summary() of all trades, the longs and the shorts, one row each
    stats.daily()                 # per session: gross / net PnL, cumulative, drawdown
    stats.rolling()               # per session: the last month's (21 sessions) gross / net PnL, Sharpe, win rate
    stats.by_period('month')      # per calendar month (or 'year')
    stats.by_label('hour')        # per value of a label of the signal bar: session, rth, hour, news_1
    BacktestStats.grid(trades)    # summary() of every (tp, sl) pair

Gross is before costs (`gross_usd`), net after slippage and commission (`net_usd`). Anything sequential (equity,
drawdown, streaks) goes in exit order. First draft by Gemini, reviewed and extended.
"""
import math

import numpy as np
import polars as pl

NAN = float('nan')


def _peak(col: str) -> pl.Expr:
    """Running peak of a cumulative PnL column, starting at 0 (so a loss from the very start is a drawdown)."""
    return pl.max_horizontal(pl.col(col).cum_max(), pl.lit(0.0))


def _longest_run(flags: np.ndarray) -> int:
    """Longest run of consecutive True values."""
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


class BacktestStats:
    """Statistics of one backtest (one TP / SL pair); grid() handles several stacked in one table."""

    def __init__(self, trades: pl.DataFrame, sessions=None, periods_per_year: int = 252, month_sessions: int = 21):
        """
        Args:
            trades: add_costs' output for one backtest.
            sessions: every session date in the tested range; a session without a trade is a 0 $ day in the daily
                series (Sharpe, Sortino, positive sessions, rolling months). None: only the days with trades.
            periods_per_year: sessions a year, to annualise Sharpe and Sortino.
            month_sessions: sessions in a rolling month.
        """
        self.trades = trades.sort(['exit_ts', 'entry_ts'])
        self.sessions = None if sessions is None else pl.Series('date', list(sessions), dtype=pl.Date).unique().sort()
        self.periods_per_year = periods_per_year
        self.month_sessions = month_sessions

    def daily(self) -> pl.DataFrame:
        """
        One row per session: trades, wins, gross_usd, net_usd, cum_gross_usd, cum_net_usd and drawdown_usd (net
        cumulative minus its running peak, the peak starting at 0: always <= 0).
        """
        days = self.trades.group_by('date').agg(
            trades=pl.len(),
            wins=(pl.col('net_usd') > 0).sum(),
            gross_usd=pl.col('gross_usd').sum(),
            net_usd=pl.col('net_usd').sum(),
        )
        if self.sessions is not None:
            days = self.sessions.to_frame().join(days, on='date', how='left').with_columns(pl.exclude('date').fill_null(0))
        return days.sort('date').with_columns(
            cum_gross_usd=pl.col('gross_usd').cum_sum(),
            cum_net_usd=pl.col('net_usd').cum_sum(),
        ).with_columns(drawdown_usd=pl.col('cum_net_usd') - _peak('cum_net_usd'))

    def rolling(self, window: int | None = None) -> pl.DataFrame:
        """
        Per session, over the last `window` sessions (default a month, month_sessions): gross and net PnL
        (`gross_usd_{w}`, `net_usd_{w}`), the annualised Sharpe of the daily net (`sharpe_{w}`) and the trades' win
        rate (`win_rate_{w}`). Null until `window` sessions have passed.
        """
        w = window or self.month_sessions
        net = pl.col('net_usd')
        return self.daily().select(
            'date',
            pl.col('gross_usd').rolling_sum(w).alias(f'gross_usd_{w}'),
            net.rolling_sum(w).alias(f'net_usd_{w}'),
            (net.rolling_mean(w) / net.rolling_std(w) * math.sqrt(self.periods_per_year)).alias(f'sharpe_{w}'),
            (pl.col('wins').rolling_sum(w) / pl.col('trades').rolling_sum(w)).alias(f'win_rate_{w}'),
        )

    def summary(self) -> dict:
        """
        One backtest in numbers. Win = net_usd > 0. Drawdown is trade by trade on the net; its length is the longest
        run of sessions below the peak. Sharpe / Sortino are on the daily net, annualised. worst_month_usd and
        pct_positive_months are over rolling months (month_sessions). With no trades: counts 0, ratios NaN.
        """
        t = self.trades
        n = len(t)
        net = t['net_usd'].to_numpy()
        gross = t['gross_usd'].to_numpy()
        wins, losses = net[net > 0], net[net < 0]
        result = t['result'].to_numpy()
        long = t['side'].to_numpy() == 1

        d = self.daily()
        day_net = d['net_usd'].to_numpy()
        std = day_net.std(ddof=1) if len(day_net) > 1 else 0.0
        downside = math.sqrt(np.mean(np.minimum(day_net, 0.0) ** 2)) if len(day_net) else 0.0
        ann = math.sqrt(self.periods_per_year)
        months = self.rolling()[f'net_usd_{self.month_sessions}'].drop_nulls().to_numpy()
        cum = np.cumsum(net)
        drawdown = cum - np.maximum.accumulate(np.maximum(cum, 0.0))

        def share(x):
            return x / n if n else NAN

        return {
            'trades': n,
            'longs': int(long.sum()),
            'shorts': int((~long).sum()),
            'win_rate': share(len(wins)),
            'tp_share': share(int((result == 1).sum())),
            'sl_share': share(int((result == -1).sum())),
            'end_share': share(int((result == 0).sum())),
            'avg_win_usd': wins.mean() if len(wins) else NAN,
            'avg_loss_usd': losses.mean() if len(losses) else NAN,
            'avg_gross_usd': gross.mean() if n else NAN,
            'avg_net_usd': net.mean() if n else NAN,
            'total_gross_usd': float(gross.sum()),
            'total_cost_usd': float(gross.sum() - net.sum()),
            'total_net_usd': float(net.sum()),
            'gross_long_usd': float(gross[long].sum()),
            'gross_short_usd': float(gross[~long].sum()),
            'net_long_usd': float(net[long].sum()),
            'net_short_usd': float(net[~long].sum()),
            'profit_factor': wins.sum() / -losses.sum() if len(losses) else (math.inf if len(wins) else NAN),
            'max_drawdown_usd': float(drawdown.min()) if n else 0.0,
            'max_drawdown_sessions': _longest_run(d['drawdown_usd'].to_numpy() < 0),
            'sharpe': day_net.mean() / std * ann if std > 0 else NAN,
            'sortino': day_net.mean() / downside * ann if downside > 0 else NAN,
            'pct_positive_sessions': float((day_net > 0).mean()) if len(day_net) else NAN,
            'worst_month_usd': float(months.min()) if len(months) else NAN,
            'pct_positive_months': float((months > 0).mean()) if len(months) else NAN,
            'longest_losing_streak': _longest_run(net < 0),
            # exit_ts is the start of the exit bar: holding time to the minute
            'avg_hold_minutes': (t['exit_ts'] - t['entry_ts']).dt.total_seconds().to_numpy().mean() / 60 if n else NAN,
        }

    def for_side(self, side: int) -> 'BacktestStats':
        """The same statistics on the longs (side 1) or the shorts (-1) only, over the same sessions."""
        return BacktestStats(self.trades.filter(pl.col('side') == side), self.sessions, self.periods_per_year,
                             self.month_sessions)

    def by_side(self) -> pl.DataFrame:
        """summary() of all trades, the longs only and the shorts only: one row each, `side` = all / long / short."""
        parts = {'all': self, 'long': self.for_side(1), 'short': self.for_side(-1)}
        return pl.DataFrame([{'side': name, **stats.summary()} for name, stats in parts.items()])

    def by_period(self, period: str = 'month') -> pl.DataFrame:
        """
        Per calendar 'year' or 'month' (a Date, the period's first day): trades, win rate, gross / net PnL and the net
        drawdown within the period (trade level, from the period's start). Periods without trades are left out.
        """
        every = {'year': '1y', 'month': '1mo'}[period]
        return (
            self.trades.with_columns(period=pl.col('date').dt.truncate(every))
            .with_columns(_cum=pl.col('net_usd').cum_sum().over('period'))
            .group_by('period')
            .agg(
                trades=pl.len(),
                win_rate=(pl.col('net_usd') > 0).mean(),
                gross_usd=pl.col('gross_usd').sum(),
                net_usd=pl.col('net_usd').sum(),
                max_drawdown_usd=(pl.col('_cum') - _peak('_cum')).min(),
            )
            .sort('period')
        )

    def by_label(self, col: str) -> pl.DataFrame:
        """
        Per value of a label column of the trades (the entry bar's `session`, `rth`, `hour` or `news_1`): trades, win rate,
        gross and net PnL, also split long / short, and the average net per trade.
        """
        long = pl.col('side') == 1
        return self.trades.group_by(col).agg(
            trades=pl.len(),
            win_rate=(pl.col('net_usd') > 0).mean(),
            gross_usd=pl.col('gross_usd').sum(),
            net_usd=pl.col('net_usd').sum(),
            gross_long_usd=pl.col('gross_usd').filter(long).sum(),
            gross_short_usd=pl.col('gross_usd').filter(~long).sum(),
            net_long_usd=pl.col('net_usd').filter(long).sum(),
            net_short_usd=pl.col('net_usd').filter(~long).sum(),
            avg_net_usd=pl.col('net_usd').mean(),
        ).sort(col)

    @staticmethod
    def grid(trades: pl.DataFrame, by=('tp', 'sl'), sessions=None, **kwargs) -> pl.DataFrame:
        """summary() of each `by` group (each its own backtest) as one row, `by` columns first, best total net first."""
        rows = [{**dict(zip(by, key)), **BacktestStats(group, sessions, **kwargs).summary()}
                for key, group in trades.group_by(list(by))]
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows).sort('total_net_usd', descending=True, nulls_last=True)
