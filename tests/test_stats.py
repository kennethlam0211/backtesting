# tests/test_stats.py
import math
from datetime import date
from datetime import datetime as dt

import polars as pl
import pytest

from backtesting.backtest import add_costs, one_at_a_time
from backtesting.plots import BacktestPlots
from backtesting.stats import BacktestStats

SCHEMA = {
    "row": pl.UInt32, "signal_ts": pl.Datetime("ms"), "date": pl.Date, "side": pl.Int8,
    "tp": pl.Int32, "sl": pl.Int32, "entry_ts": pl.Datetime("ms"), "entry_ind": pl.Int64,
    "entry_px": pl.Int64, "exit_row": pl.UInt32, "exit_ts": pl.Datetime("ms"),
    "exit_px": pl.Int64, "result": pl.Int8, "pnl": pl.Int64,
}
DEF = {"row": 0, "signal_ts": dt(2024, 1, 2, 9, 30), "date": date(2024, 1, 2), "side": 1,
       "tp": 10, "sl": 10, "entry_ts": dt(2024, 1, 2, 9, 30), "entry_ind": 0,
       "entry_px": 100, "exit_row": 1, "exit_ts": dt(2024, 1, 2, 9, 31),
       "exit_px": 100, "result": 0, "pnl": 0}
LDT = {"rth": pl.Boolean, "hour": pl.Int64}


def mk(*trades, **labels):
    df = pl.DataFrame({k: [t.get(k, v) for t in trades] for k, v in DEF.items()}, schema=SCHEMA)
    for k, v in labels.items():
        df = df.with_columns(pl.Series(k, v, dtype=LDT.get(k, pl.Int8)))
    return df


def T(d, h, m):
    return dt(2024, 1, d, h, m)


D2, D3, D4 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
MAIN = mk(  # holds: 15, 30, 10, 60 min; nets -40, 10, 80, -20 (gross == net here)
    dict(date=D2, side=-1, tp=10, sl=20, entry_ts=T(2, 9, 30), exit_ts=T(2, 9, 45), exit_px=140, result=-1, pnl=-40),
    dict(date=D2, side=-1, tp=5, sl=5, entry_ts=T(2, 10, 0), exit_ts=T(2, 10, 30), exit_px=90, result=1, pnl=10),
    dict(date=D4, side=1, tp=10, sl=20, entry_ts=T(4, 9, 31), exit_ts=T(4, 9, 41), exit_px=180, result=1, pnl=80),
    dict(date=D4, side=1, tp=5, sl=5, entry_ts=T(4, 13, 0), exit_ts=T(4, 14, 0), exit_px=80, result=0, pnl=-20),
    session=[0, 1, 0, 1],
)
SESS = [D2, D3, D4]  # D3 has no trades
CST = add_costs(MAIN, 1.0, 0.0, 0.0)  # net_usd == gross_usd == pnl
ST = BacktestStats(CST, sessions=SESS, month_sessions=1)
EMPTY = CST.head(0)
KEYS = ["trades", "longs", "shorts", "win_rate", "tp_share", "sl_share", "end_share",
        "avg_win_usd", "avg_loss_usd", "avg_gross_usd", "avg_net_usd", "total_gross_usd",
        "total_cost_usd", "total_net_usd", "gross_long_usd", "gross_short_usd",
        "net_long_usd", "net_short_usd", "profit_factor", "max_drawdown_usd",
        "max_drawdown_sessions", "sharpe", "sortino", "pct_positive_sessions",
        "worst_month_usd", "pct_positive_months", "longest_losing_streak", "avg_hold_minutes"]


def test_one_at_a_time():
    tr = mk(
        dict(row=0, exit_row=3), dict(row=1, exit_row=4), dict(row=2, exit_row=2),
        dict(row=3, exit_row=9), dict(row=8, exit_row=10),
    )
    kept = one_at_a_time(tr)
    assert kept.columns == tr.columns
    # keep row0(exit 3); skip row1(<3) and row2(<3, its exit 2 must not block); keep row3(==3); skip row8(<9)
    assert kept["row"].to_list() == [0, 3]
    assert kept["exit_row"].to_list() == [3, 9]


def test_add_costs():
    tr = mk(
        dict(side=1, entry_px=100, exit_px=150, result=1, pnl=50),
        dict(side=-1, entry_px=200, exit_px=180, result=-1, pnl=-20),
        dict(side=1, entry_px=100, exit_px=90, result=0, pnl=-10),
    )
    out = add_costs(tr, 2.0, 1.5, 3)
    assert out.columns[:14] == list(SCHEMA)
    assert out["gross_usd"].to_list() == pytest.approx([100.0, -40.0, -20.0])
    assert out["net_units"].to_list() == [47, -26, -16]  # 50-3, -20-2*3, -10-2*3
    assert out["net_usd"].to_list() == pytest.approx([91.0, -55.0, -35.0])  # units*2 - 2*1.5
    assert out["net_units"].dtype == pl.Int64 and out["gross_usd"].dtype == pl.Float64
    z = add_costs(tr, 1.0, 0.0, 0.0)
    assert z["net_usd"].to_list() == pytest.approx(z["gross_usd"].to_list())
    assert z["net_usd"].to_list() == pytest.approx([50.0, -20.0, -10.0])


def test_daily_and_rolling():
    d = ST.daily()
    assert d["date"].to_list() == SESS
    assert d["trades"].to_list() == [2, 0, 2] and d["wins"].to_list() == [1, 0, 1]  # D3 zero day
    assert d["net_usd"].to_list() == pytest.approx([-30.0, 0.0, 60.0])
    assert d["cum_gross_usd"].to_list() == pytest.approx([-30.0, -30.0, 30.0])
    assert d["cum_net_usd"].to_list() == pytest.approx([-30.0, -30.0, 30.0])
    assert d["drawdown_usd"].to_list() == pytest.approx([-30.0, -30.0, 0.0])  # first trade loses
    r = ST.rolling(window=2)
    assert r.columns == ["date", "gross_usd_2", "net_usd_2", "sharpe_2", "win_rate_2"]
    assert r["net_usd_2"].to_list() == [None, pytest.approx(-30.0), pytest.approx(60.0)]
    assert r["sharpe_2"].to_list() == [None, pytest.approx(-11.2249721603),  # -15/sqrt(450)*sqrt(252) = -sqrt(126)
                                       pytest.approx(11.2249721603)]
    assert r["win_rate_2"].to_list() == [None, pytest.approx(0.5), pytest.approx(0.5)]
    assert ST.rolling()["net_usd_1"].to_list() == pytest.approx([-30.0, 0.0, 60.0])


def test_summary():
    sm = ST.summary()
    assert list(sm) == KEYS
    exp = {
        "trades": 4, "longs": 2, "shorts": 2, "win_rate": 0.5,
        "tp_share": 0.5, "sl_share": 0.25, "end_share": 0.25,
        "avg_win_usd": 45.0, "avg_loss_usd": -30.0, "avg_gross_usd": 7.5, "avg_net_usd": 7.5,
        "total_gross_usd": 30.0, "total_cost_usd": 0.0, "total_net_usd": 30.0,
        "gross_long_usd": 60.0, "gross_short_usd": -30.0, "net_long_usd": 60.0, "net_short_usd": -30.0,
        "profit_factor": 1.5,  # (10+80) / (40+20)
        "max_drawdown_usd": -40.0,  # trade cums -40,-30,50,30 -> min dd -40
        "max_drawdown_sessions": 2,  # daily dd [-30,-30,0]
        "sharpe": pytest.approx(3.4641016151377544),  # 10/sqrt(2100)*sqrt(252)
        "sortino": pytest.approx(9.1651513899),  # downside sqrt((900+0+0)/3) = sqrt(300): 10/sqrt(300)*sqrt(252)
        "pct_positive_sessions": pytest.approx(1 / 3),
        "worst_month_usd": -30.0, "pct_positive_months": pytest.approx(1 / 3),  # month_sessions=1
        "longest_losing_streak": 1, "avg_hold_minutes": 28.75,  # (15+30+10+60)/4
    }
    for k, v in exp.items():
        assert sm[k] == v


def test_by_side_period_label_grid():
    bs = ST.by_side()
    assert bs["side"].to_list() == ["all", "long", "short"]
    assert bs.columns[1:] == KEYS
    rows = {r["side"]: r for r in bs.iter_rows(named=True)}
    assert rows["all"]["total_net_usd"] == pytest.approx(30.0)
    assert rows["long"]["profit_factor"] == pytest.approx(4.0)  # 80 / 20
    assert rows["short"]["profit_factor"] == pytest.approx(0.25)  # 10 / 40
    assert (rows["short"]["trades"], rows["short"]["total_net_usd"]) == (2, pytest.approx(-30.0))
    fs = ST.for_side(-1).summary()
    assert (fs["trades"], fs["shorts"], fs["total_net_usd"]) == (2, 2, pytest.approx(-30.0))
    per = ST.by_period("month")
    assert per["period"].to_list() == [date(2024, 1, 1)]
    p0 = per.row(0, named=True)
    assert (p0["trades"], p0["win_rate"], p0["net_usd"]) == (4, pytest.approx(0.5), pytest.approx(30.0))
    assert p0["max_drawdown_usd"] == pytest.approx(-40.0)
    assert ST.by_period("year")["period"].to_list() == [date(2024, 1, 1)]
    bl = ST.by_label("session")
    assert bl[bl.columns[0]].to_list() == [0, 1]
    b0, b1 = bl.iter_rows(named=True)
    assert (b0["net_usd"], b0["gross_long_usd"], b0["gross_short_usd"], b0["avg_net_usd"]) == (
        pytest.approx(40.0), pytest.approx(80.0), pytest.approx(-40.0), pytest.approx(20.0))
    assert (b1["net_usd"], b1["avg_net_usd"]) == (pytest.approx(-10.0), pytest.approx(-5.0))
    g = BacktestStats.grid(CST)
    assert g.columns[:2] == ["tp", "sl"] and g["tp"].to_list() == [10, 5]
    assert g["total_net_usd"].to_list() == pytest.approx([40.0, -10.0])  # sorted desc


def test_zero_trades():
    st0 = BacktestStats(EMPTY, sessions=[D2])
    sm = st0.summary()  # must not raise
    assert sm["trades"] == 0 and sm["max_drawdown_usd"] == 0.0
    assert math.isnan(sm["sharpe"]) and math.isnan(sm["profit_factor"])
    assert st0.daily().height == 1  # the session row, zeros
    assert BacktestStats.grid(EMPTY).height == 0


def test_plots(tmp_path):
    paths = BacktestPlots(ST, tmp_path, title="t").all()
    assert [p.name for p in paths] == ["equity.png", "rolling_month.png", "pnl_by_year.png",
                                       "pnl_by_month.png", "trade_pnl.png"]
    for p in paths:
        assert p.parent == tmp_path and p.stat().st_size > 0
    hp = BacktestPlots.grid_heatmap(BacktestStats.grid(CST), "total_net_usd", tmp_path)
    assert hp == tmp_path / "grid_total_net_usd.png" and hp.stat().st_size > 0
    neg = BacktestStats.grid(add_costs(mk(
        dict(side=-1, entry_px=100, exit_px=140, result=-1, pnl=-40, tp=10, sl=20),
        dict(side=1, entry_px=100, exit_px=80, result=0, pnl=-20, tp=5, sl=5)), 1.0, 0.0, 0.0))
    hn = BacktestPlots.grid_heatmap(neg, "total_net_usd", tmp_path)
    assert hn.stat().st_size > 0  # all-negative values
