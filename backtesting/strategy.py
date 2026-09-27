"""
Strategies for backtesting/backtest.py. A Strategy names the training-data columns it reads and adds `side`
(1 long, -1 short, 0 none) to read_training_data's frame: a state, 1 or -1 on every bar while its condition holds,
known at the bar's close. ensemble() joins them into one state, and find_exits picks the entries from it
(config.SEARCH_MODE) and holds each trade from the next bar.

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
    side: int = 0                                    # the side it signals: 1 long only, -1 short only, 0 either


def _rsi(freq: str) -> pl.Expr:
    """
    The RSI of `freq` bars, 0-100: feature_engineering's `{WINDOW_SIZE}_sma_rsi_{freq}` (e.g. 20_sma_rsi_1), RSI(14)
    of the close changes with simple 14-bar means (Cutler's RSI, not Wilder's smoothing), stored as -(RSI / 50 - 1)
    in [-1, 1]. On FREQ rows a larger freq's RSI is its last closed bar's.
    """
    return (1 - pl.col(f'{WINDOW_SIZE}_sma_rsi_{freq}')) / 2 * 100


def _state(name: str, freq: str, cond: pl.Expr, side: int, reverse: bool, **params) -> Strategy:
    """
    `side` while `cond` holds, else 0; reverse: the other side. Named {name}_{freq}_{params' values}, plus _reversed.
    """
    side = -side if reverse else side
    return Strategy('_'.join([name, freq, *(str(v) for v in params.values())]) + ('_reversed' if reverse else ''),
                    (f'{WINDOW_SIZE}_sma_rsi_{freq}',),
                    lambda df: df.with_columns(side=pl.when(cond).then(side).otherwise(0).cast(pl.Int8)),
                    params={'freq': freq, **params, 'reverse': reverse}, side=side)


def rsi_long(freq: str, level: float, reverse: bool = False) -> Strategy:
    """
    Mean reversion: 1 while the RSI of `freq` bars is below `level`, else 0 (rsi_long_{freq}_{level}). reverse:
    -1 instead, momentum (rsi_long_{freq}_{level}_reversed).
    """
    return _state('rsi_long', freq, _rsi(freq) < level, 1, reverse, level=level)


def rsi_short(freq: str, level: float, reverse: bool = False) -> Strategy:
    """
    Mean reversion: -1 while the RSI of `freq` bars is above `level`, else 0 (rsi_short_{freq}_{level}). reverse:
    1 instead, momentum (rsi_short_{freq}_{level}_reversed).
    """
    return _state('rsi_short', freq, _rsi(freq) > level, -1, reverse, level=level)


def expand(signals: dict) -> list[Strategy]:
    """
    One Strategy per factory x freq x combination of its params' lists (config.SIGNALS). A factory takes one grid or
    a list of them (e.g. other levels on another freq).
    """
    return [factory(freq=f, **dict(zip(grid['params'], combo)))
            for factory, grids in signals.items()
            for grid in (grids if isinstance(grids, list) else [grids])
            for f in grid['freq']
            for combo in itertools.product(*grid['params'].values())]


def ensemble(strategies: list[Strategy], rule: Callable, side: int | None = None, search_mode: bool = False) -> Strategy:
    """
    One Strategy from many: each strategy's `side` becomes a column named after it, then
    `rule(df, signals, search_mode)` (e.g. backtesting/ensemble.py's ensemble_function; `signals`: each column's name
    -> the side it signals) turns them into `side`, a state. Named after the rule; each trade keeps the signal columns
    of its signal bar, and `signals`, a label for the analysis: the position (L long, S short) and one digit per
    signal of its side in `strategies` order, 1 where it is active (e.g. 'L110100...').

    side: 1 or -1: the rule sees that side's signals only (and either-side ones) and its state keeps that side only,
        so longs and shorts run as separate strategies (`{rule}_long` / `{rule}_short`): neither blocks nor closes
        the other's trades.
    """
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError(f"strategy names must be unique: {names}")

    def signals(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([s.signals(df)['side'].alias(s.name) for s in strategies])
        vote = rule(df, {s.name: s.side for s in strategies if s.name in own}, search_mode=search_mode)
        state = (vote if side is None else pl.when(vote == side).then(vote).otherwise(0)).cast(pl.Int8)
        position = pl.when(state == 1).then(pl.lit('L')).when(state == -1).then(pl.lit('S')).otherwise(pl.lit(''))
        return df.with_columns(side=state, signals=pl.concat_str(
            [position, *[(pl.col(n) != 0).cast(pl.Int8).cast(pl.Utf8) for n in own]]))

    own = [s.name for s in strategies if side is None or s.side in (side, 0)]  # the side's signals
    columns = tuple(dict.fromkeys(c for s in strategies for c in s.columns))
    name = getattr(rule, '__name__', 'ensemble') + {None: '', 1: '_long', -1: '_short'}[side]
    return Strategy(name, columns, signals, keep=(*names, 'signals'), side=side or 0)
