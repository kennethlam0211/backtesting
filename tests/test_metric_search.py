"""backtesting/metric_search.py: the stop-grid neighbours, the ranks and the per-condition picks."""
import numpy as np
import polars as pl

from backtesting import metric_search as ms


def test_neighbours_are_one_grid_step_apart_both_ways():
    runs = pl.DataFrame({'signal': ['a'] * 6 + ['b'],
                         'stops': ['16/16', '16/24', '24/16', '60/60', '1x/1x', '2x/1x', '16/16']}).with_row_index('k')
    A = ms.neighbours(runs).toarray().astype(int)
    assert (A == A.T).all()  # the grid steps are signed: 1 - 2 must not wrap around
    assert A[0].tolist() == [1, 1, 1, 0, 0, 0, 0]  # 16/16: itself, 16/24, 24/16; not 60/60 (two steps away)
    assert A[3].tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert A[4].tolist() == [0, 0, 0, 0, 1, 1, 0]  # vol stops only next to vol stops
    assert A[6].tolist() == [0, 0, 0, 0, 0, 0, 1]  # another signal is never a neighbour


def test_pct_rank_puts_nan_last_and_averages_ties():
    np.testing.assert_allclose(ms.pct_rank(np.array([3.0, np.nan, 1.0, 3.0])), [0.875, 0.25, 0.5, 0.875])


def test_pick_takes_the_best_run_per_condition_then_the_top():
    class B:  # the one attribute pick reads
        cond = np.array([0, 0, 1, 2])
    idx = np.array([0, 1, 2, 3])
    ranks = {'sharpe': np.array([0.9, 1.0, 0.2, 0.5])}
    assert ms.pick(B, ranks, idx, {'sharpe': 1.0}, 2).tolist() == [1, 3]  # run 0 loses to run 1, its condition's best
