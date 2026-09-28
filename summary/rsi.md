# RSI strategies with limit entries

Research summary, 2026-09-28. ES futures, 2020-05 to 2026-09-18 (6.1 years). Each strategy trades 1 contract.

**Result in one line:** with limit orders throughout (no slippage), most RSI strategies still lose. A handful beat
random entries, and pairing a long with a short roughly doubles the Calmar. The best pair is **buy + sell when 60-min
RSI > 85**, with Calmar 1.92 over the whole period and 2.10 before 2026. Five such pairs together reach Calmar 4.37
(see [Best 5](#best-5-a-portfolio-of-same-condition-pairs), with how to reproduce it). Everything here is a best case
until the stricter fill rule is in (see [Caveats](#caveats)).

## Units, names, costs

- Prices are in units of 0.25 index point; 1 unit = $12.50. Stops are written TP/SL in units: `16/100` is a 4-point
  take-profit and a 25-point stop-loss.
- **Costs: $1.90 commission per side, no slippage.** A limit run uses limit orders throughout: the entry, the
  take-profit, the stop-loss as a stop-limit at its level, and the signal and 22:58 exits as limits at the next bar's
  open. (A market run still pays 1 unit per market fill.)
- Signal names come from `backtesting/strategy.py`. `rsi_long_10_15` means "buy while 10-min RSI < 15";
  `rsi_short_60_85` means "sell while 60-min RSI > 85". `_reversed` trades the other side, so
  `rsi_short_60_85_reversed` means "**buy** while 60-min RSI > 85". The tables below use the plain wording.

## The limit entry model

A limit order at the entry bar's open, sent at the bar's start, reaches the market 300 ms later (the trading
machine's latency).

- **Better price on arrival** (below the open for a buy, above it for a sell): fills at once, at that price.
- **Worse price on arrival:** waits for price to come back to the open before the minute ends, and fills at the
  open; otherwise there is no trade.
- **Skipped** if the TP or SL level (measured from the open) was hit during the 300 ms or while the order waited.

All of this is precomputed per minute in `1_ohlcv.parquet` by `data_pipeline/to_dat.py`; `backtesting/entry.py`
looks it up by `start_ind`. Details: `docs/to_dat.md`, `docs/backtesting.md`.

Over all 2,373,007 minutes, a buy limit at the open:

| On arrival (300 ms in) | Minutes | Result |
|---|---|---|
| Price at the open | 1,811,497 | filled at the open |
| Price below the open (better) | 283,213 | filled at that price |
| Price above the open, came back | 244,339 | filled at the open |
| Price above the open, never came back | 33,906 | skipped |
| No trade after 300 ms | 52 | skipped |

98.6% filled: 87.9% of fills at the open and 12.1% at a better price. The sell side is the mirror image.

## The sweep

- Trade bar sizes: 5, 10, 15, 30 and 60-min. The RSI is on the same bar size as the trade.
- RSI levels: 1/99, 2/98, 3/97, 5/95, 10/90 and 15/85. Normal and reversed, longs and shorts as separate strategies.
- Stops: all 36 pairs of 16, 24, 32, 40, 60 and 100 units.
- Settings: normal mode (one trade per signal), volatility band (0.2, 0.8), no entries on news.
- 110 signals with trades × 36 stops = 3,960 runs. The whole grid takes about 7 minutes.

### Cost models compared

Over the 1,152 runs with at least 100 trades:

| | Market entry, 1 unit per market fill | Limit entry, 1 unit on SL / signal / flat exits | **Limit orders throughout, no slippage** |
|---|---|---|---|
| Positive runs | 221 | 407 | **511** |
| Median net per trade | −$22.9 | −$9.9 | **−$3.1** |

Limit entries keep 97.8% of the trades. Even with no slippage, the typical run loses.

## Drift check

Each run is compared with 50 random entries on the same side, with the same stops, the same limit fills and the same
costs. The random entries are placed in the **same month and hour of day** as the run's trades. p is the share of
random runs that did at least as well; 0.02 is the lowest possible with 50 runs.

An earlier version matched on the same *day* and hour. That let a random entry land before the signal in its hour and
catch the move that caused it, which made mean-reversion longs look good and momentum shorts look bad. Don't use it.

Random entries over all hours show no bias in the backtest: random longs gain a little from ES's rise and random
shorts lose the same.

## Best single strategies by Calmar

Calmar = net $ per year ÷ max drawdown, trade by trade on the net, over the whole period, runs with at least 100
trades, best stops per signal.

| Side | Signal | Stops | Trades | $/trade | Max DD | Calmar | 2026 $/trade | p |
|---|---|---|---|---|---|---|---|---|
| Sell | 60-min RSI > 85 | 16/60 | 204 | +57.7 | $1,844 | 1.05 | +62.4 | **0.04** |
| Sell | 10-min RSI < 10 | 24/60 | 146 | +47.1 | $2,078 | 0.54 | +59.2 | 0.06 |
| Sell | 10-min RSI < 15 | 60/24 | 400 | +48.6 | $6,522 | 0.49 | +152.0 | **0.02** |
| Sell | 15-min RSI < 15 | 100/32 | 284 | +51.4 | $6,582 | 0.36 | +220.8 | 0.08 |
| Sell | 30-min RSI < 15 | 16/40 | 154 | +26.3 | $3,109 | 0.21 | −89.8 | 0.08 |
| Buy | 5-min RSI > 90 | 60/32 | 392 | +66.0 | $3,832 | 1.11 | +1.8 | **0.02** |
| Buy | 60-min RSI > 85 | 16/100 | 202 | +79.4 | $2,523 | 1.04 | +204.5 | **0.02** |
| Buy | 30-min RSI > 90 | 60/24 | 104 | +105.2 | $2,722 | 0.66 | +122.9 | **0.04** |
| Buy | 30-min RSI > 85 | 40/16 | 318 | +29.5 | $2,361 | 0.65 | +19.3 | **0.04** |
| Buy | 10-min RSI > 85 | 40/24 | 670 | +24.3 | $4,813 | 0.55 | +24.6 | 0.18 |

- The best single Calmar is about 1: a year's profit roughly equals the worst drawdown.
- 60-min RSI > 85 works as both a buy and a sell with a 4-point TP. The edge is a calm, oscillating market, not
  direction.
- Every pick buys strength or sells weakness (momentum), except the 60-min RSI > 85 sell.

## Pairs: one long + one short

Every profitable long run (329) with every profitable short run (148): 48,692 pairs. Their daily P&L is summed and
ranked by Calmar on the daily curve. The long and short legs are almost uncorrelated day to day, so their drawdowns
don't line up.

**Best pair, whichever period it's ranked on: buy + sell when 60-min RSI > 85** (buy 16/32, sell 16/60).

| | Calmar | Max drawdown | Net per year | 2026 |
|---|---|---|---|---|
| Whole period (6.1 years) | **1.92** (rank 1) | $1,664 | $3,201 | +$679, max DD $923 |
| Before 2026 (5.4 years) | **2.10** (rank 1) | $1,664 | | |

Each leg alone has a Calmar of 0.41 (buy) and 1.05 (sell).

Next best over the whole period:

| Rank | Long | Short | Calmar | Max DD | Net per year | 2026 net | 2026 max DD |
|---|---|---|---|---|---|---|---|
| 2 | buy 60-min RSI > 85, 16/40 | sell 60-min RSI > 85, 16/60 | 1.66 | $1,894 | $3,151 | +$979 | $828 |
| 3 | buy 5-min RSI > 90, 60/40 | sell 60-min RSI > 85, 16/100 | 1.61 | $3,986 | $6,399 | +$2,055 | $3,786 |
| 4 | buy 5-min RSI > 90, 60/32 | sell 10-min RSI < 10, 24/60 | 1.55 | $3,472 | $5,365 | +$1,431 | $3,423 |
| 5 | buy 60-min RSI > 85, 16/100 | sell 10-min RSI < 10, 24/40 | 1.53 | $2,502 | $3,833 | +$4,267 | $504 |
| 6 | buy 60-min RSI > 85, 16/100 | sell 10-min RSI < 10, 24/24 | 1.51 | $2,282 | $3,439 | +$3,067 | $672 |

- Ranking on 2020–2025 predicts 2026 only weakly: of the top 15 pairs by Calmar before 2026, 6 made money in 2026
  (5 of 15 with same-condition pairs excluded). Pairs using "sell when 30-min RSI < 15" lost $0.8k–5k in 2026.
- Ranks 5 and 6 pair two different signals and had the best 2026 (+$3.1k to +$4.3k, max drawdown about $500–700).
  They are the fallback if the same-open double fill of the best pair doesn't hold.

## Best 5: a portfolio of same-condition pairs

Five pairs, each one RSI condition traded both ways (same bar size and level; only the side and the stops differ), 10
strategies at 1 contract each. Built from the best pair by adding, one condition at a time, the pair that raises the
portfolio's Calmar most (whole period), while it rises by at least 0.02. Only legs that are profitable on their own
count (`--positive-legs`).

| Step | Pair added (condition) | Long | Short | Portfolio Calmar | Before 2026 | Max DD | Net per year | 2026 net |
|---|---|---|---|---|---|---|---|---|
| 1 | 60-min RSI > 85 (`rsi_short_60_85`) | buy 16/32 | sell 16/60 | 1.92 | 2.10 | $1,664 | $3,201 | +$679 |
| 2 | 30-min RSI < 15 (`rsi_long_30_15`) | buy 60/32 | sell 40/60 | 2.60 | 2.82 | $1,574 | $4,100 | +$1,133 |
| 3 | 15-min RSI < 15 (`rsi_long_15_15`) | buy 40/32 | sell 24/24 | 3.22 | 3.43 | $1,949 | $6,272 | +$2,251 |
| 4 | 10-min RSI < 10 (`rsi_long_10_10`) | buy 32/60 | sell 40/40 | 4.16 | 4.45 | $2,100 | $8,735 | +$2,938 |
| **5** | **15-min RSI > 90 (`rsi_short_15_90`)** | **buy 16/24** | **sell 16/16** | **4.37** | **4.71** | **$2,050** | **$8,969** | **+$2,675** |

Each pair on its own: Calmar 1.92, 0.41, 0.75, 1.21 and 0.17 in the order above. The weak ones (30-min RSI < 15,
15-min RSI > 90) are in for their timing: their drawdowns fall when the others' do not, so they smooth the total more
than they add profit. **Picked on the whole period, 2026 included**, so 2026 does not test it; data after
2026-09-18 will. All legs rely on limit fills at the open for a buy and a sell at once, the fill model's most
optimistic case (see [Caveats](#caveats)).

### This version's plots

Kept with this summary, so later runs cannot overwrite them (the same files are rebuilt in `results/sweep/limit/`):

- [rsi/top5_pairs.png](rsi/top5_pairs.png): the best pair of each condition, top 5, each with its buy and sell legs
  (`acc_pnl_pairs.png`).
- [rsi/best5_portfolio.png](rsi/best5_portfolio.png): the best 5 portfolio and each pair in one panel
  (`acc_pnl_portfolio.png`).
- [rsi/best5_portfolio_pairs.png](rsi/best5_portfolio_pairs.png): the best 5 portfolio on top, then one panel per pair
  with its long and short legs (`acc_pnl_portfolio_pairs.png`).

### How to reproduce it

Needs step 2's `tick.dat` and `1_ohlcv.parquet` with the limit-entry columns, and step 3's training data (see
`docs/to_dat.md`, `docs/backtesting.md`). Costs and filters come from `backtesting/config.py` as they are now:
limit orders with no slippage, $1.90 commission per side, the volatility band (0.2, 0.8) over 60 sessions, no entry
on news, normal mode, flat at 22:58.

```bash
python -m backtesting.sweep rsi --out results/sweep/limit        # ~7 min: every RSI run, trades per bar size
python -m backtesting.pairs limit --positive-legs --portfolio     # the pairs, the best 5, plots and statistics
```

The second command prints the table above (Calmar 1.92, 2.60, 3.22, 4.16, 4.37) and writes to `results/sweep/limit/`:
`portfolio.parquet` (the steps), `acc_pnl_portfolio.png` (the portfolio and each pair's curve in one panel: the
09:24 plot), `acc_pnl_portfolio_pairs.png` (the portfolio, then one panel per pair with its long and short legs),
plus `pairs.parquet` and `acc_pnl_pairs.png` (the best pair of each condition, top 5: the 09:52 plot).
Verified on 2026-09-28: the sweep gives identical trades to the original run; rerunning `pairs.py` gives pixel-identical
plots (the three above) and identical tables.

## Portfolio: adding strategies one at a time

Starting from the best pair, each step adds the run (either side, one per signal) that raises the Calmar before 2026
most, stopping when it no longer rises by 0.02.

| Strategies | Added | Calmar before 2026 | Max DD | Net per year | 2026 net | 2026 max DD |
|---|---|---|---|---|---|---|
| 1 | buy 60-min RSI > 85, 16/32 | 0.41 | $3,486 | $1,430 | +$54 | $574 |
| 2 | + sell 60-min RSI > 85, 16/60 | 2.10 | $1,664 | $3,499 | +$679 | $923 |
| **3** | **+ sell 10-min RSI < 10, 24/24** | **2.45** | **$1,760** | **$4,305** | **+$1,291** | **$728** |
| 4 | + sell 30-min RSI < 15, 16/40 | 3.06 | $1,746 | $5,341 | −$236 | $2,654 |
| 5 | + buy 30-min RSI < 15, 16/24 | 3.11 | $1,811 | $5,632 | +$1,446 | $1,184 |
| 6 | + buy 10-min RSI < 10, 24/60 | 3.47 | $2,090 | $7,245 | −$2,059 | $2,694 |
| 7 | + buy 5-min RSI < 10, 16/16 | 3.99 | $1,896 | $7,569 | −$3,654 | $4,463 |
| 8 | + sell 5-min RSI < 10, 32/16 | 4.08 | $1,962 | $8,000 | −$2,930 | $3,511 |
| 9 | + buy 30-min RSI > 90, 16/24 | 4.13 | $2,070 | $8,549 | −$2,691 | $4,082 |

**Stop at 3.** Each added strategy raises the 2020–2025 Calmar, but from the 4th on 2026 swings and mostly gets worse
(negative from the 6th): that is fitting the past, not an edge.

## Caveats

- **The fill rule is optimistic.** It counts a trade at the open as a fill for both the buy and the sell, but only one
  side is marketable at a time; the other waits in the queue. On 20 recent days, a stricter rule using the raw
  trade-side column fills about 89% of minutes against our 98%. The missed 10% tend to be winners. Small-TP strategies,
  and above all the same-condition pair (a buy and a sell at the same open), are the most exposed.
- **No slippage assumes every exit fills at its price.** A stop-limit can miss when price jumps past the level, and a
  limit at the next bar's open can miss too; the backtest assumes they fill.
- **Selection.** These are the best of about 4,000 single runs and 49,000 pairs. A p of 0.02, or a high Calmar before
  2026, is a lead, not proof.
- **Few trades.** About 17 to 110 trades a year per strategy.
- **Opposite positions.** The two legs of a pair can be open at once. In one account they net out; the P&L is the
  same, since it's linear, but it relies on both fills happening.

## Next steps

1. Build the stricter fill rule: keep the raw trade-side column in step 1 (about 45 minutes to rerun) and change the
   `to_dat.py` entry kernel. Then rerun the pairs above.
2. Not swept yet: 1-min bars, a higher timeframe's RSI traded on lower bars, several RSI signals combined, volatility
   stops (`vol_grid`), the volatility band off, search mode, and UD with limit entries.

## Files

Under `results/sweep/` (local, not in git):

- `limit/summary_{freq}.parquet`, `limit/trades_{freq}.parquet`: every run and trade, limit orders, no slippage.
  `limit_exit_slippage/` is the same with 1 unit on stop-loss, signal and flat exits; `market/` with market entries.
- `limit/calmar.parquet`: every run's Calmar and max drawdown. `limit/pairs_all.parquet`: every pair.
- Plots in `limit/`: `acc_pnl_calmar.png` (the Calmar picks above), `acc_pnl_pairs.png` (the best pair of each
  condition, top 5), `acc_pnl_portfolio.png` (the best 5 portfolio in one panel), `acc_pnl_portfolio_pairs.png` (the
  same, one panel per pair).

In the repo: `backtesting/sweep.py` (the sweep), `backtesting/pairs.py` (pairs and the best 5 portfolio) and
`backtesting/walk_forward.py`. Still only in the session's temporary scratchpad: the drift check (`drift_limit.py`),
the single-run Calmar ranking and 3-strategy portfolio above (`calmar.py`, `portfolio.py`), and `fill_realism.py`,
`random_bias.py`.
