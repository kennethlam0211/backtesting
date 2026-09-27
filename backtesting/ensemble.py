"""
Ensemble functions for config.ENSEMBLE: `(df, signals, search_mode) -> side`, where df holds one signal column per
strategy (1 / -1 while its condition holds, else 0) and `signals` maps each column's name to the side it signals
(1 long only, -1 short only, 0 either). The result is a state too (1 / -1 while it holds); find_exits picks the
entries from it by config.SEARCH_MODE.
"""
import polars as pl


def ensemble_function(df: pl.DataFrame, signals: dict[str, int], search_mode: bool = False) -> pl.Expr:
    """
    Search mode, lenient: 1 while any long signal is 1, -1 while any short signal is -1; which fired is the trades'
    `signals` label, for the analysis to find the patterns that pay. Normal mode, strict: 1 while every long signal
    is 1, -1 while every short signal is -1. 0 when neither or both. A signal of either side counts for both; a side
    with no signals never trades.
    """
    agree = pl.any_horizontal if search_mode else pl.all_horizontal
    longs = [name for name, side in signals.items() if side >= 0]
    shorts = [name for name, side in signals.items() if side <= 0]
    long = agree(pl.col(longs) == 1) if longs else pl.lit(False)
    short = agree(pl.col(shorts) == -1) if shorts else pl.lit(False)
    return pl.when(long & ~short).then(1).when(short & ~long).then(-1).otherwise(0).cast(pl.Int8)
