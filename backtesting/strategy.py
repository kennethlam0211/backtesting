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


def pivot_range(freq: str, pivots: tuple[int, int] = (1, 2)) -> tuple[pl.Expr, pl.Expr]:
    """
    The low and high of `freq`'s U/D pivots last{pivots[0]}..last{pivots[1]} (1 = the newest; (1, 2): the last
    swing, (1, 5): all five), from `{WINDOW_SIZE}_UD_last{k}_{freq}`.
    """
    cols = [pl.col(f'{WINDOW_SIZE}_UD_last{k}_{freq}') for k in range(pivots[0], pivots[1] + 1)]
    return pl.min_horizontal(cols), pl.max_horizontal(cols)


def range_target(freq: str, pivots: tuple[int, int], reach: float) -> tuple[pl.Expr, pl.Expr]:
    """
    The take-profit levels in `freq`'s pivot range (pivot_range): `reach` of the way across it, up from the low for a
    long, down from the high for a short (0.9: 10% of the range short of the far pivot).
    """
    low, high = pivot_range(freq, pivots)
    return low + reach * (high - low), high - reach * (high - low)


def ud_range(freq: str, range_freq: str, pivots: tuple[int, int] = (1, 2), level: float = 0.618,
             trend: tuple[str, ...] | None = None, once: bool = False, skip_hit: bool = False,
             reach: float = 1.0) -> Strategy:
    """
    Inside the range of `range_freq`'s pivots (pivot_range), on `freq`'s trend (its U/D flag): 1 while `freq`'s close
    is in the range at or below its `level` retracement from the top (high - level x (high - low)) and the flag is 1
    (up); -1 while at or above low + level x (high - low) and the flag is -1. Its take-profit is the gap to the other
    end of the range (stops.gap_stops). ud_range_{freq}_{range_freq}_{pivots}_{level}.
    trend: the timeframes whose U/D flags must all point the trade's way (e.g. ('5',) or ('5', '15')), instead of
        `freq`'s own (_t5, _t5-15); `freq` stays the bar screened and traded.
    once: one signal per range and side, on the first bar the condition holds while the pivots stay the same; the next
        waits for a new range (_once).
    skip_hit: no signal when the previous `freq` bar already reached the target (range_target with `reach`, as
        stops.gap_stops' take-profit: give both the same `reach`): its high for a long, its low for a short (_skiphit).
    """
    low, high = pivot_range(range_freq, pivots)
    close = pl.col(f'close_{freq}')
    flags = [f'{WINDOW_SIZE}_UD_flag_{f}' for f in (trend or (freq,))]
    long = (close >= low) & (close <= high - level * (high - low)) & pl.all_horizontal([pl.col(c) == 1 for c in flags])
    short = (close <= high) & (close >= low + level * (high - low)) & pl.all_horizontal([pl.col(c) == -1 for c in flags])
    range_cols = [f'{WINDOW_SIZE}_UD_last{k}_{range_freq}' for k in range(pivots[0], pivots[1] + 1)]
    bar_cols = []
    if skip_hit:
        bar_cols = [f'high_{freq}', f'low_{freq}']
        long_target, short_target = range_target(range_freq, pivots, reach)
        long = long & ~(pl.col(f'high_{freq}').shift(1) >= long_target).fill_null(False)
        short = short & ~(pl.col(f'low_{freq}').shift(1) <= short_target).fill_null(False)
    if once:
        # A range: the bars in a row whose pivots do not change (the same values later are a new range)
        range_id = pl.any_horizontal([pl.col(c) != pl.col(c).shift() for c in range_cols]).fill_null(True).cum_sum()
        long = long & (long.cast(pl.Int32).cum_sum().over(range_id) == 1)
        short = short & (short.cast(pl.Int32).cum_sum().over(range_id) == 1)
    columns = (f'close_{freq}', *flags, *range_cols, *bar_cols)
    name = (f'ud_range_{freq}_{range_freq}_{pivots[0]}{pivots[1]}_{level}' + (f"_t{'-'.join(trend)}" if trend else '')
            + ('_once' if once else '') + ('_skiphit' if skip_hit else ''))
    return Strategy(name, columns,
                    lambda df: df.with_columns(side=pl.when(long).then(1).when(short).then(-1).otherwise(0).cast(pl.Int8)),
                    params={'freq': freq, 'range_freq': range_freq, 'pivots': f'{pivots[0]}-{pivots[1]}', 'level': level,
                            'trend': '-'.join(trend or (freq,)), 'once': once, 'skip_hit': skip_hit, 'reach': reach})


def ud_cross(freq: str, range_freq: str, pivots: tuple[int, int] = (1, 2), entry: float = 0.7,
             trend: tuple[str, ...] | None = None, once: bool = True, reach: float | None = None) -> Strategy:
    """
    Continuation inside the range of `range_freq`'s pivots (pivot_range): 1 on the `freq` bar whose close crosses up
    through `entry` of the range (low + entry x (high - low)) with the `trend` timeframes' U/D flags up (default
    `freq`'s own); -1 on the bar whose close crosses down through high - entry x (high - low) with them down. Its
    take-profit is a bit further on (stops.gap_stops with reach > entry, e.g. 0.75).
    ud_cross_{freq}_{range_freq}_{pivots}_{entry}.
    once: one signal per range and side (the next waits for new pivots).
    reach: the take-profit level (as stops.gap_stops'); a cross only counts while the close is still short of it.
    """
    low, high = pivot_range(range_freq, pivots)
    close = pl.col(f'close_{freq}')
    up, down = low + entry * (high - low), high - entry * (high - low)
    flags = [f'{WINDOW_SIZE}_UD_flag_{f}' for f in (trend or (freq,))]
    range_cols = [f'{WINDOW_SIZE}_UD_last{k}_{range_freq}' for k in range(pivots[0], pivots[1] + 1)]
    # A range: the bars in a row whose pivots do not change; a cross only counts inside one
    range_id = pl.any_horizontal([pl.col(c) != pl.col(c).shift() for c in range_cols]).fill_null(True).cum_sum()
    same = range_id == range_id.shift()
    long_end, short_end = range_target(range_freq, pivots, reach) if reach is not None else (high, low)
    long = (same & (close.shift() < up) & (close >= up) & (close < long_end)
            & pl.all_horizontal([pl.col(c) == 1 for c in flags])).fill_null(False)
    short = (same & (close.shift() > down) & (close <= down) & (close > short_end)
             & pl.all_horizontal([pl.col(c) == -1 for c in flags])).fill_null(False)
    if once:
        long = long & (long.cast(pl.Int32).cum_sum().over(range_id) == 1)
        short = short & (short.cast(pl.Int32).cum_sum().over(range_id) == 1)
    name = (f'ud_cross_{freq}_{range_freq}_{pivots[0]}{pivots[1]}_{entry}' + (f"_t{'-'.join(trend)}" if trend else '')
            + ('_once' if once else ''))
    return Strategy(name, (f'close_{freq}', *flags, *range_cols),
                    lambda df: df.with_columns(side=pl.when(long).then(1).when(short).then(-1).otherwise(0).cast(pl.Int8)),
                    params={'freq': freq, 'range_freq': range_freq, 'pivots': f'{pivots[0]}-{pivots[1]}', 'entry': entry,
                            'trend': '-'.join(trend or (freq,)), 'once': once, 'reach': reach})


def ud_near(freq: str, pivot_freq: str, dist: int, pivots: tuple[int, int] = (1, 5), side: int = 1) -> Strategy:
    """
    Near a pivot: `side` (1 long, -1 short) while `freq`'s close is within `dist` units of any of `pivot_freq`'s U/D
    pivots last{pivots[0]}..last{pivots[1]} (1 = the newest), else 0. Meant to run on both sides, each with its own
    stops (ud_near_{freq}_{pivot_freq}_{pivots}_{dist}_long / _short).
    """
    cols = [f'{WINDOW_SIZE}_UD_last{k}_{pivot_freq}' for k in range(pivots[0], pivots[1] + 1)]
    close = pl.col(f'close_{freq}')
    near = pl.any_horizontal([(close - pl.col(c)).abs() <= dist for c in cols]).fill_null(False)
    name = f"ud_near_{freq}_{pivot_freq}_{pivots[0]}{pivots[1]}_{dist}_{'long' if side == 1 else 'short'}"
    return Strategy(name, (f'close_{freq}', *cols),
                    lambda df: df.with_columns(side=pl.when(near).then(side).otherwise(0).cast(pl.Int8)),
                    params={'freq': freq, 'pivot_freq': pivot_freq, 'pivots': f'{pivots[0]}-{pivots[1]}', 'dist': dist,
                            'side': side}, side=side)


def ud_end(freq: str, pivot_freq: str, dist: int, end: str = 'high', side: int = 1) -> Strategy:
    """
    Near one end of the range: `side` while `freq`'s close is within `dist` units of the higher (end 'high') or the
    lower (end 'low') of `pivot_freq`'s last two U/D pivots, else 0; nothing until both pivots exist
    (ud_end_{freq}_{pivot_freq}_{end}_{dist}_long / _short). ud_near with pivots (1, 2) is both ends at once.
    """
    if end not in ('high', 'low'):
        raise ValueError(f"end must be 'high' or 'low', not {end!r}")
    a, b = (pl.col(f'{WINDOW_SIZE}_UD_last{k}_{pivot_freq}') for k in (1, 2))
    level = pl.max_horizontal(a, b) if end == 'high' else pl.min_horizontal(a, b)
    close = pl.col(f'close_{freq}')
    near = (((close - level).abs() <= dist) & a.is_not_null() & b.is_not_null()).fill_null(False)
    name = f"ud_end_{freq}_{pivot_freq}_{end}_{dist}_{'long' if side == 1 else 'short'}"
    return Strategy(name, (f'close_{freq}', f'{WINDOW_SIZE}_UD_last1_{pivot_freq}', f'{WINDOW_SIZE}_UD_last2_{pivot_freq}'),
                    lambda df: df.with_columns(side=pl.when(near).then(side).otherwise(0).cast(pl.Int8)),
                    params={'freq': freq, 'pivot_freq': pivot_freq, 'end': end, 'dist': dist, 'side': side}, side=side)


def ud_level(freq: str, pivot_freq: str, dist: int, frac: float, swings: tuple[int, int] = (1, 1),
             side: int = 1) -> Strategy:
    """
    Near a level inside a swing: `side` while `freq`'s close is within `dist` units of the level `frac` of the way from
    a swing's newer pivot back to its older one (last{k} + frac * (last{k+1} - last{k}): 0.5 its middle, 0.3 a 30%
    retracement) for any swing k in swings[0]..swings[1] of `pivot_freq`'s U/D pivots (swing 1: last1 to last2, the
    range), else 0 (ud_level_{freq}_{pivot_freq}_{swings}_{frac %}_{dist}_long / _short).
    """
    ks = range(swings[0], swings[1] + 1)
    cols = [f'{WINDOW_SIZE}_UD_last{k}_{pivot_freq}' for k in range(swings[0], swings[1] + 2)]
    close = pl.col(f'close_{freq}')
    pivot = lambda k: pl.col(f'{WINDOW_SIZE}_UD_last{k}_{pivot_freq}')
    near = pl.any_horizontal([(close - (pivot(k) + frac * (pivot(k + 1) - pivot(k)))).abs() <= dist
                              for k in ks]).fill_null(False)
    name = (f"ud_level_{freq}_{pivot_freq}_{swings[0]}{swings[1]}_{round(frac * 100)}_{dist}_"
            f"{'long' if side == 1 else 'short'}")
    return Strategy(name, (f'close_{freq}', *cols),
                    lambda df: df.with_columns(side=pl.when(near).then(side).otherwise(0).cast(pl.Int8)),
                    params={'freq': freq, 'pivot_freq': pivot_freq, 'swings': f'{swings[0]}-{swings[1]}', 'frac': frac,
                            'dist': dist, 'side': side}, side=side)


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
