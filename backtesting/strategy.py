"""
Strategies for backtesting/backtest.py. A Strategy names the training-data columns it reads and adds `side`
(1 long, -1 short, 0 none) to read_training_data's frame; a signal is known at its bar's close, and find_exits
enters on the next bar. The backtest runs the ones listed in backtesting/config.py over the TP x SL grid.

To add one: write its signal function and a factory returning a Strategy, then list it in config.STRATEGIES.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl

from params import WINDOW_SIZE


@dataclass(frozen=True)
class Strategy:
    name: str                                        # results folder and chart titles
    columns: tuple[str, ...]                         # training-data columns signals() reads
    signals: Callable[[pl.DataFrame], pl.DataFrame]  # adds `side`
    params: dict = field(default_factory=dict)       # its settings, written on each of its trades as columns


def rsi_signals(df: pl.DataFrame, low: float | None, high: float | None, freq: str) -> pl.DataFrame:
    """
    Mean reversion on the RSI of `freq` bars: long on the bar the RSI first goes below `low`, short on the bar it
    first goes above `high`. Bars that stay inside the zone do not signal again. None: that side does not trade.

    The RSI is feature_engineering's `{WINDOW_SIZE}_sma_rsi_{freq}` (e.g. 20_sma_rsi_1): RSI(14) of the close changes with simple 14-bar means
    (Cutler's RSI, not Wilder's smoothing), stored as -(RSI / 50 - 1) in [-1, 1]. It is turned back into 0-100 and
    kept as `rsi_{freq}`.
    """
    rsi = pl.col(f'rsi_{freq}')
    long = (rsi < low) & (rsi.shift(1) >= low) if low is not None else pl.lit(False)
    short = (rsi > high) & (rsi.shift(1) <= high) if high is not None else pl.lit(False)
    return df.with_columns(((1 - pl.col(f'{WINDOW_SIZE}_sma_rsi_{freq}')) / 2 * 100).alias(f'rsi_{freq}')).with_columns(
        side=pl.when(long).then(1).when(short).then(-1).otherwise(0).cast(pl.Int8)
    )


def rsi(low: float | None = None, high: float | None = None, *, freq: str) -> Strategy:
    """
    RSI mean reversion (rsi_signals) on `freq` bars: long below `low` and / or short above `high`. A long depends only
    on `low`, a short only on `high`, so one side alone is the natural unit: rsi_long_{low}, rsi_short_{high}.
    """
    name = f'rsi_long_{low}' if high is None else f'rsi_short_{high}' if low is None else f'rsi_{low}_{high}'
    return Strategy(name, (f'{WINDOW_SIZE}_sma_rsi_{freq}',), lambda df: rsi_signals(df, low, high, freq),
                    params={'rsi_low': low, 'rsi_high': high})
