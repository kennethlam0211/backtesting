"""
Stop functions for config.STOP: `(signals) -> signals` with `tp` and `sl` (units, > 0) and `stops`, the run label.
find_exits calls it on the signal rows (every loaded column), which must stay in row order; each `stops` value is
its own backtest (one trade at a time). A function that sets no `stops` is labelled with its name.
"""
import polars as pl

from backtesting.strategy import range_target


def grid(tp: list[int], sl: list[int]):
    """Every signal with every (tp, sl) pair, all in one walk; `stops` = '{tp}/{sl}'."""
    levels = pl.DataFrame([(a, b) for a in tp for b in sl], schema={'tp': pl.Int64, 'sl': pl.Int64}, orient='row')

    def grid_stops(signals: pl.DataFrame) -> pl.DataFrame:
        return (signals.drop(['tp', 'sl'], strict=False).join(levels, how='cross')
                .with_columns(stops=pl.format('{}/{}', 'tp', 'sl')))

    return grid_stops


def vol_grid(tp: list[float], sl: list[float], column: str):
    """
    Every signal with every (tp, sl) pair of multiples of its signal bar's volatility `column` (units, e.g. config's
    VOL_COLUMN), all in one walk: wider in a volatile market, tighter in a quiet one. tp / sl = ceil(k x volatility),
    at least 1 unit; `stops` = '{tp}x/{sl}x'.
    """
    levels = pl.DataFrame([(a, b, f'{a}x/{b}x') for a in tp for b in sl],
                          schema={'_tp_k': pl.Float64, '_sl_k': pl.Float64, 'stops': pl.Utf8}, orient='row')

    def vol_stops(signals: pl.DataFrame) -> pl.DataFrame:
        level = lambda k: (pl.col(k) * pl.col(column)).ceil().clip(lower_bound=1).cast(pl.Int64)
        return (signals.drop(['tp', 'sl', 'stops'], strict=False).join(levels, how='cross')
                .with_columns(tp=level('_tp_k'), sl=level('_sl_k')).drop('_tp_k', '_sl_k'))

    return vol_stops


def gap_stops(range_freq: str, pivots: tuple[int, int], price: str, ratios: list[float], reach: float = 1.0):
    """
    Fill the gap (with strategy.ud_range): the take-profit is the distance from the signal bar's `price` (e.g.
    close_1) to the level `reach` of the way across `range_freq`'s pivot range toward its far end
    (strategy.range_target; 0.9: 10% of the range short of the pivot, which price may not get back to). The
    stop-loss each of `ratios` x the take-profit. At least 1 unit each; `stops` = 'gap{reach}/{ratio}x'.
    """
    long_target, short_target = range_target(range_freq, pivots, reach)
    levels = pl.DataFrame({'_r': [float(r) for r in ratios], 'stops': [f'gap{reach}/{r}x' for r in ratios]})

    def gap_stops(signals: pl.DataFrame) -> pl.DataFrame:
        tp = pl.when(pl.col('side') == 1).then(long_target - pl.col(price)).otherwise(pl.col(price) - short_target)
        return (signals.drop(['tp', 'sl', 'stops'], strict=False).join(levels, how='cross')
                .with_columns(tp=tp.ceil().clip(lower_bound=1).cast(pl.Int64),
                              sl=(tp * pl.col('_r')).ceil().clip(lower_bound=1).cast(pl.Int64)).drop('_r'))

    return gap_stops
