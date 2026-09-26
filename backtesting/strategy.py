"""
Strategies for backtesting/backtest.py. A Strategy names the training-data columns it reads and adds `side`
(1 long, -1 short, 0 none) to read_training_data's frame; a signal is known at its bar's close, and find_exits
enters on the next bar. The backtest runs the ones listed in backtesting/config.py over the TP x SL grid.

To add one: write its signal function and a factory returning a Strategy, then list it in config.STRATEGIES.
"""
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Strategy:
    name: str                                        # results folder and chart titles
    columns: tuple[str, ...]                         # training-data columns signals() reads
    signals: Callable[[pl.DataFrame], pl.DataFrame]  # adds `side`


def rsi_signals(df: pl.DataFrame, low: float, high: float) -> pl.DataFrame:
    """
    Mean reversion on the 1-min RSI: long on the bar the RSI first goes below `low`, short on the bar it first goes
    above `high`. Bars that stay inside the zone do not signal again.

    The RSI is feature_engineering's `20_sma_rsi_1`: RSI(14) of the 1-min close changes with simple 14-bar means
    (Cutler's RSI, not Wilder's smoothing), stored as -(RSI / 50 - 1) in [-1, 1]. It is turned back into 0-100 and
    kept as `rsi_1`.
    """
    rsi = pl.col('rsi_1')
    return df.with_columns(rsi_1=(1 - pl.col('20_sma_rsi_1')) / 2 * 100).with_columns(
        side=pl.when((rsi < low) & (rsi.shift(1) >= low)).then(1)
        .when((rsi > high) & (rsi.shift(1) <= high)).then(-1)
        .otherwise(0)
        .cast(pl.Int8)
    )


def rsi(low: float, high: float) -> Strategy:
    """1-min RSI mean reversion (rsi_signals) at levels low / high."""
    return Strategy(f'rsi_{low}_{high}', ('20_sma_rsi_1',), lambda df: rsi_signals(df, low, high))
