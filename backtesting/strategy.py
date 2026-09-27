"""
Strategies for backtesting/backtest.py. A Strategy names the training-data columns it reads and adds `side`
(1 long, -1 short, 0 none) to read_training_data's frame: a state, 1 or -1 on every bar while its condition holds,
known at the bar's close. ensemble() joins them; its `side` is 1 / -1 only on the bar its rule turns 1 / -1, and
find_exits holds the trade from the next bar.

config.SIGNALS lists factories with a `freq` list and a `params` dict of lists; expand() makes one Strategy per
combination and ensemble() joins them into one `side` with config.ENSEMBLE. To add one: write a factory
`(freq, **params) -> Strategy` and list it in config.SIGNALS.
"""
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl

from params import WINDOW_SIZE


@dataclass(frozen=True)
class Strategy:
    name: str                                        # the `strategy` column of its trades, chart titles
    columns: tuple[str, ...]                         # training-data columns signals() reads
    signals: Callable[[pl.DataFrame], pl.DataFrame]  # adds `side`
    params: dict = field(default_factory=dict)       # its settings, written on each of its trades as columns
    keep: tuple[str, ...] = ()                       # columns signals() adds that each trade keeps from its signal bar


def _rsi(freq: str) -> pl.Expr:
    """
    The RSI of `freq` bars, 0-100: feature_engineering's `{WINDOW_SIZE}_sma_rsi_{freq}` (e.g. 20_sma_rsi_1), RSI(14)
    of the close changes with simple 14-bar means (Cutler's RSI, not Wilder's smoothing), stored as -(RSI / 50 - 1)
    in [-1, 1]. On FREQ rows a larger freq's RSI is its last closed bar's.
    """
    return (1 - pl.col(f'{WINDOW_SIZE}_sma_rsi_{freq}')) / 2 * 100


def _state(name: str, freq: str, cond: pl.Expr, side: int, **params) -> Strategy:
    return Strategy(f"{name}_{freq}_{'_'.join(str(v) for v in params.values())}".rstrip('_'),
                    (f'{WINDOW_SIZE}_sma_rsi_{freq}',),
                    lambda df: df.with_columns(side=pl.when(cond).then(side).otherwise(0).cast(pl.Int8)),
                    params={'freq': freq, **params})


def rsi_long(freq: str, level: float) -> Strategy:
    """Mean reversion: 1 while the RSI of `freq` bars is below `level`, else 0 (rsi_long_{freq}_{level})."""
    return _state('rsi_long', freq, _rsi(freq) < level, 1, level=level)


def rsi_short(freq: str, level: float) -> Strategy:
    """Mean reversion: -1 while the RSI of `freq` bars is above `level`, else 0 (rsi_short_{freq}_{level})."""
    return _state('rsi_short', freq, _rsi(freq) > level, -1, level=level)


def expand(signals: dict) -> list[Strategy]:
    """One Strategy per factory x freq x combination of its params' lists (config.SIGNALS)."""
    return [factory(freq=f, **dict(zip(grid['params'], combo)))
            for factory, grid in signals.items()
            for f in grid['freq']
            for combo in itertools.product(*grid['params'].values())]


def ensemble(strategies: list[Strategy], rule: Callable) -> Strategy:
    """
    One Strategy from many: each strategy's `side` becomes a column named after it, then `rule(df, names)` (e.g.
    backtesting/ensemble.py's ensemble_function) turns them into a state. `side` is that state on the bar it turns
    1 / -1 (the trigger bar), else 0. Named after the rule; each trade keeps the signal columns of its signal bar.
    """
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError(f"strategy names must be unique: {names}")

    def signals(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([s.signals(df)['side'].alias(s.name) for s in strategies])
        state = pl.col('_state').fill_null(0)
        return df.with_columns(_state=rule(df, names)).with_columns(
            side=pl.when((state != 0) & (state != state.shift(fill_value=0))).then(state).otherwise(0).cast(pl.Int8)
        ).drop('_state')

    columns = tuple(dict.fromkeys(c for s in strategies for c in s.columns))
    return Strategy(getattr(rule, '__name__', 'ensemble'), columns, signals, keep=tuple(names))
