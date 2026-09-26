"""
Charts of a backtest (BacktestStats), saved as PNG files (matplotlib, no display needed):

    plots = BacktestPlots(stats, out_dir, title='RSI 10/90, TP 40 SL 24')
    plots.all()                                                    # every chart below, returns the paths
    BacktestPlots.grid_heatmap(grid, 'total_net_usd', out_dir)     # one statistic over the TP x SL grid

Colors follow the repo's chart palette: categorical hues in a fixed order (net = slot 1 blue, gross = slot 2
orange), diverging blue / red around 0 for PnL signs, solid hairline y gridlines, text in ink colors (never a series
color), one y axis per chart. First draft by Gemini, reviewed and extended.
"""
import pathlib

import matplotlib

matplotlib.use('Agg')  # files only, no display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

# Chart chrome and ink
BG = '#fcfcfb'
TEXT_PRI = '#0b0b0b'
TEXT_SEC = '#52514e'
TEXT_MUT = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
# Categorical slots, in order: net / take-profit, gross / stop-loss, session end
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']  # fixed order
SLOT = CATEGORICAL[:3]
NET, GROSS = SLOT[0], SLOT[1]
# Diverging pair around 0
BLUE, RED, NEUTRAL = '#2a78d6', '#e34948', '#f0efec'
DIVERGING = LinearSegmentedColormap.from_list('pnl', [RED, NEUTRAL, BLUE])
NEGATIVE = LinearSegmentedColormap.from_list('pnl_neg', [RED, NEUTRAL])   # one arm, when every value is <= 0
POSITIVE = LinearSegmentedColormap.from_list('pnl_pos', [NEUTRAL, BLUE])  # ... or >= 0


def _usd(x) -> str:
    return f'-${-x:,.0f}' if x < 0 else f'${x:,.0f}'


USD = ticker.FuncFormatter(lambda x, _: _usd(x))


def _style(ax, ylabel=None, usd=True):
    """The shared chrome: surface background, y hairlines behind the data, only the bottom spine, muted ticks."""
    ax.set_facecolor(BG)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(axis='both', colors=TEXT_MUT, length=0, pad=6)
    ax.grid(axis='y', color=GRID, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_MUT, fontsize=10)
    if usd:
        ax.yaxis.set_major_formatter(USD)


def _dates(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _legend(ax):
    # Above the plot, right-aligned: never over the data
    leg = ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 1), ncol=2, fontsize=9)
    for text in leg.get_texts():
        text.set_color(TEXT_SEC)


def _save(fig, path: pathlib.Path) -> pathlib.Path:
    fig.patch.set_facecolor(BG)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    return path


class BacktestPlots:
    """The charts of one backtest; out_dir is created if missing."""

    def __init__(self, stats, out_dir, title: str = ''):
        self.stats = stats
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.title = title

    def _title(self, name: str) -> str:
        return f'{self.title} · {name}' if self.title else name

    def all(self) -> list[pathlib.Path]:
        """Every chart of this backtest."""
        return [self.equity(), self.rolling(), self.by_period('year'), self.by_period('month'), self.trade_distribution()]

    def equity(self) -> pathlib.Path:
        """
        equity.png: cumulative gross and net PnL by session for all trades, the longs and the shorts (small multiples on
        one $ scale), and the net drawdown of all trades below.
        """
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, gridspec_kw={'height_ratios': [2, 2, 2, 1.3]})
        fig.suptitle(self._title('Cumulative PnL'), color=TEXT_PRI, fontsize=13, x=0.01, ha='left')
        parts = [('All trades', self.stats), ('Long', self.stats.for_side(1)), ('Short', self.stats.for_side(-1))]
        for ax, (name, stats) in zip(axes, parts):
            d = stats.daily()
            if not d.is_empty():
                dates = d['date'].to_list()
                ax.plot(dates, d['cum_gross_usd'].to_numpy(), color=GROSS, linewidth=1.5, label='Gross', zorder=3)
                ax.plot(dates, d['cum_net_usd'].to_numpy(), color=NET, linewidth=1.5, label='Net', zorder=3)
            ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
            ax.set_title(name, color=TEXT_SEC, fontsize=10, loc='left')
            _style(ax)
        axes[1].sharey(axes[0])
        axes[2].sharey(axes[0])
        _legend(axes[0])

        dd = self.stats.daily()
        ax = axes[3]
        if not dd.is_empty():
            dates = dd['date'].to_list()
            ax.fill_between(dates, dd['drawdown_usd'].to_numpy(), 0, color=RED, alpha=0.35, linewidth=0, zorder=2)
            ax.plot(dates, dd['drawdown_usd'].to_numpy(), color=RED, linewidth=1.5, zorder=3)
        ax.set_title('Drawdown, net, all trades', color=TEXT_SEC, fontsize=10, loc='left')
        _style(ax)
        _dates(ax)
        fig.tight_layout()
        return _save(fig, self.out_dir / 'equity.png')

    def rolling(self) -> pathlib.Path:
        """rolling_month.png: gross and net PnL over the last month (month_sessions sessions), every session."""
        w = self.stats.month_sessions
        r = self.stats.rolling()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.set_title(self._title(f'Rolling month PnL ({w} sessions)'), color=TEXT_PRI, fontsize=12, loc='left', pad=12)
        if not r.is_empty():
            dates = r['date'].to_list()
            ax.plot(dates, r[f'gross_usd_{w}'].to_numpy(), color=GROSS, linewidth=1.5, label='Gross', zorder=3)
            ax.plot(dates, r[f'net_usd_{w}'].to_numpy(), color=NET, linewidth=1.5, label='Net', zorder=3)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        _style(ax)
        _dates(ax)
        _legend(ax)
        fig.tight_layout()
        return _save(fig, self.out_dir / 'rolling_month.png')

    def by_period(self, period: str = 'year') -> pathlib.Path:
        """pnl_by_{period}.png: net PnL per calendar year or month, positive bars blue and negative red."""
        df = self.stats.by_period(period)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.set_title(self._title(f'Net PnL by {period}'), color=TEXT_PRI, fontsize=12, loc='left', pad=12)
        if not df.is_empty():
            net = df['net_usd'].to_numpy()
            width = np.timedelta64(24 if period == 'month' else 300, 'D')  # ~0.8 of the period: the surface is the gap
            ax.bar(df['period'].to_list(), net, width=width, color=np.where(net >= 0, BLUE, RED), edgecolor='none', zorder=3)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        _style(ax)
        _dates(ax)
        fig.tight_layout()
        return _save(fig, self.out_dir / f'pnl_by_{period}.png')

    def trade_distribution(self) -> pathlib.Path:
        """trade_pnl.png: histogram of net PnL per trade, one panel per exit type, on a shared $ axis."""
        # Shared $ axis; each panel its own count axis (take-profits can outnumber time exits 100 to 1)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        fig.suptitle(self._title('Net PnL per trade, by exit'), color=TEXT_PRI, fontsize=12, x=0.01, ha='left')
        trades = self.stats.trades
        # One set of bins for the three panels: with fixed TP / SL a panel can hold a single value, which per-panel
        # bins would squeeze into an invisible sliver
        edges = np.histogram_bin_edges(trades['net_usd'].to_numpy(), bins=40) if len(trades) else 40
        for ax, (code, name, color) in zip(axes, [(1, 'Take-profit', SLOT[0]), (-1, 'Stop-loss', SLOT[1]),
                                                  (0, 'Time exit (22:59) / close', SLOT[2])]):
            subset = trades.filter(pl.col('result') == code)['net_usd'].to_numpy()
            if len(subset):
                ax.hist(subset, bins=edges, color=color, edgecolor='none', rwidth=0.8, zorder=3)
            ax.set_title(f'{name} ({len(subset):,})', color=TEXT_SEC, fontsize=10, loc='left')
            _style(ax, usd=False)
            ax.xaxis.set_major_formatter(USD)
        axes[0].set_ylabel('trades', color=TEXT_MUT, fontsize=10)
        fig.tight_layout()
        return _save(fig, self.out_dir / 'trade_pnl.png')

    @staticmethod
    def grid_heatmap(grid: pl.DataFrame, value: str, out_dir, by=('tp', 'sl'), title: str = '') -> pathlib.Path:
        """
        grid_{value}.png: `value` (a grid() column, e.g. total_net_usd or sharpe) over the first `by` column (y) and the
        second (x), diverging around 0. Only the best cell is labelled; the grid csv is the table view.
        """
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        y_col, x_col = by
        y_vals = sorted(grid[y_col].unique().to_list())
        x_vals = sorted(grid[x_col].unique().to_list())
        matrix = np.full((len(y_vals), len(x_vals)), np.nan)
        for row in grid.iter_rows(named=True):
            matrix[y_vals.index(row[y_col]), x_vals.index(row[x_col])] = row[value]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title or value, color=TEXT_PRI, fontsize=12, loc='left', pad=12)
        finite = matrix[np.isfinite(matrix)]
        # Blue = above 0, red = below, the neutral gray at 0; when every cell has one sign only that arm is used
        cmap, norm = DIVERGING, None
        if len(finite) and finite.min() != finite.max():
            lo, hi = finite.min(), finite.max()
            if lo < 0 < hi:
                norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
            elif hi <= 0:
                cmap, norm = NEGATIVE, Normalize(vmin=lo, vmax=0)
            else:
                cmap, norm = POSITIVE, Normalize(vmin=0, vmax=hi)
        im = ax.imshow(np.where(np.isfinite(matrix), matrix, np.nan), cmap=cmap, norm=norm, origin='lower',
                       aspect='auto')
        ax.set_xticks(range(len(x_vals)), [str(v) for v in x_vals])
        ax.set_yticks(range(len(y_vals)), [str(v) for v in y_vals])
        ax.set_xlabel(x_col.upper(), color=TEXT_MUT)
        ax.set_ylabel(y_col.upper(), color=TEXT_MUT)
        ax.tick_params(colors=TEXT_MUT, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        if len(finite):
            best = np.unravel_index(np.nanargmax(np.where(np.isfinite(matrix), matrix, np.nan)), matrix.shape)
            label = _usd(matrix[best]) if value.endswith('usd') else f'{matrix[best]:.2f}'
            ax.text(best[1], best[0], label, ha='center', va='center', color=TEXT_PRI, fontsize=9, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=TEXT_MUT, length=0)
        if value.endswith('usd'):
            cbar.ax.yaxis.set_major_formatter(USD)
        return _save(fig, out_dir / f'grid_{value}.png')
