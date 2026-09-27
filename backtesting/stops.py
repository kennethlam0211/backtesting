"""
Stop functions for config.STOP: `(signals) -> signals` with `tp` and `sl` (units, > 0) and `stops`, the run label.
find_exits calls it on the signal rows (every loaded column), which must stay in row order; each `stops` value is
its own backtest (one trade at a time). A function that sets no `stops` is labelled with its name.
"""
import polars as pl


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
