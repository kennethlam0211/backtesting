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
