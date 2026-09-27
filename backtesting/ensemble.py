"""
Ensemble functions for config.ENSEMBLE: `(df, names) -> side`, where df holds one signal column per strategy (1 / -1
while its condition holds, else 0) named in `names`. The result is a state too; strategy.ensemble() keeps only the bar
it turns 1 / -1 as the signal.
"""
import polars as pl


def ensemble_function(df: pl.DataFrame, names: list[str], share: float = 0.3) -> pl.Expr:
    """
    For the time being, a vote of the active signals: with s the sum of the signals on a bar and n how many are not 0,
    1 when s >= share * n, -1 when s <= -share * n, else 0 (and 0 when none is active).
    """
    s = pl.sum_horizontal(pl.col(names).cast(pl.Int32))
    n = pl.sum_horizontal((pl.col(names) != 0).cast(pl.Int32))
    return (pl.when((n > 0) & (s >= share * n)).then(1)
            .when((n > 0) & (s <= -share * n)).then(-1)
            .otherwise(0).cast(pl.Int8))
