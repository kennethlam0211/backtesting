# Backtesting — `backtesting/`

Runs strategies over the training data (step 3's `training_data.parquet`) and decides every trade's exit on the
ticks: `stop_search` finds which of the take-profit (TP) and stop-loss (SL) a trade touches first, bar by bar.
All trades of a run go into **one table**, `results/trades.parquet`, which the analysis queries.

```bash
python -m backtesting.backtest          # config.SIGNALS through config.ENSEMBLE over config.STOP -> results/trades.parquet
python -m backtesting.analysis          # leaderboard, label analysis, the final run's statistics and charts
```

Run from the repo root. The run is set in `backtesting/config.py`; paths come from `params/params.py`.

**Units.** Prices, TP / SL and `pnl` are in units of price x 4, as in the bar files and `tick.dat`: 1 unit =
0.25 index point = $12.50 (`POINT_VALUE`). A stop pair is written `tp/sl` in units: `16/100` is a 4-point TP
and a 25-point SL.

## Inputs

| File | From | Used for |
|---|---|---|
| `data/processed/training_data.parquet` | step 3, `feature_engineering.py` | one row per 1-min bar with every timeframe's bars and features: the signals |
| `data/processed/{freq}_ohlcv.parquet` | step 2, `to_dat.py` | the traded bars when `FREQ` is above 1 min (start, labels); the 1-min file's limit-entry columns |
| `data/processed/tick.dat` | step 2 | the exits (`stop_search.StopSearch`) and the entry / exit prices |

The three must come from the same build: the backtest stops if `tick.dat`'s price at an entry is not the training
data's open there.

## Modules

| Module | What it does |
|---|---|
| `config.py` | Every setting of a run (below) |
| `backtest.py` | The backtest itself: steps 1–6 below, and `main` |
| `strategy.py` | `Strategy`, the signal factories (`rsi_long`, `rsi_short`, `ud_range`, `ud_cross`), `expand`, `ensemble` |
| `ensemble.py` | `ensemble_function`: turns the signal columns into one `side` |
| `stops.py` | Stop functions: `grid`, `vol_grid`, `gap_stops` |
| `entry.py` | Limit entries: `Entries`, `get_entry_many` |
| `analysis.py` | Queries of the trades table: leaderboard, label slices, chance check, stability, final run |
| `stats.py` | `BacktestStats`: summary, daily curve, drawdown, Sharpe, per period / label / side, the TP x SL grid |
| `plots.py` | `BacktestPlots`: equity, rolling, per period, trade distribution, grid heatmaps |
| `sweep.py` | Sweeps of a signal preset (`rsi`, `ud_near`, `ud_range`), long and short, over a stop grid (see *Research tools*) |
| `pairs.py` | Same-condition long + short pairs of a sweep, ranked by Calmar, plotted, with statistics |
| `walk_forward.py` | Walk-forward selection: each side picks its best run on past hours / days / months and trades the next |
| `ml.py`, `ml_ensemble.py`, `ml_stops.py` | A LightGBM ensemble and stop model, walk-forward (parked, see *ML*) |

## Settings (`config.py`)

| Setting | Meaning |
|---|---|
| `FREQ` | Bar size traded, any of `params.FREQS` ('1', '5', '10', '15', '30', '60', 'day'; not 'session', a feature freq with no bars in `tick.dat`): signals at its bars' close, entry at the next bar, exits walked bar by bar |
| `VOL_COLUMN` | The `FREQ` bars' volatility column, e.g. `20_std_15` (units) |
| `SIGNALS` | `{factory: {'freq': [...], 'params': {name: [values]}}}`, or a list of such grids per factory; every combination is one signal column |
| `ENSEMBLE` | A function turning the signal columns into `side` (`ensemble.ensemble_function`) |
| `SEARCH_MODE` | False: normal mode; True: search mode (see *Signals and entries*) |
| `NO_ENTRY` | Columns (0 / 1) that block an entry bar, e.g. `[f'news_{FREQ}']`; an open trade is kept |
| `VOL_BAND`, `VOL_SESSIONS` | No entry when the signal bar's `VOL_COLUMN` is outside these quantiles of the previous `VOL_SESSIONS` sessions (past only); the first `VOL_SESSIONS` sessions do not trade. `None`: no rule |
| `STOP` | A stop function giving each signal its `tp` / `sl`: `grid(tp=[...], sl=[...])` or `vol_grid(...)` |
| `START`, `END` | First / last session, inclusive; `None`: all |
| `POINT_VALUE`, `COMMISSION`, `SLIPPAGE` | $12.50 a unit; $1.90 per side; 1 unit per market fill (market runs only: a limit run pays none) |
| `FLAT_AT` | Shifted-clock time (`'22:58'`): every position closes when the bar holding it starts |
| `ENTRY` | `'market'` or `'limit'` (see *Entries*). The limit's 300 ms latency is not set here: it is `stop_search.params.ENTRY_LATENCY_MS`, built into the 1-min bar file by step 2 |
| `SPLIT` | Analysis: trades before it pick a slice, trades from it test it |
| `LABEL_FEATURES` | Training columns kept on each trade from its signal bar, bucketed into quantiles by the analysis, e.g. `{VOL_COLUMN: 5}` |
| `FINAL` | The run the analysis reports in full, `(strategy, stops)`; `None`: the best by total net |
| `ML_*` | Settings of the parked ML models |

## How a backtest runs

`main` builds one Strategy per side from `SIGNALS` and `ENSEMBLE` and runs `run_backtest` on each; the steps:

1. **`read_training_data`** — the training data as `FREQ` bars, one row each, with `date` (the session). The burn-in
   is dropped (whole sessions, until one starts with every timeframe's 5 U/D pivots). For `FREQ` above 1 min, each
   bar is its first appearance in the 1-min rows; its start and labels come from its bar file; the columns of every
   smaller timeframe are as of the bar's close, never after it. So any bar size trades without rerunning feature
   engineering.
2. **Signals** — the strategy adds `side` (see below), and `vol_ok` marks the bars inside `VOL_BAND`.
3. **`find_exits`** — every signal becomes a trade: entry on the next bar, then the bars are checked one at a time
   with `StopSearch.first_hit_many`, for every open trade at once, until a level is hit or the trade must exit.
4. **`one_at_a_time`** — per stop pair, a signal while the previous trade is still open is skipped.
5. **`add_costs`** — gross and net PnL in units and $.
6. **`run_grid`** — steps 3–5 for every stop pair of `STOP` in one walk (far fewer, larger `stop_search` calls),
   one calendar month of sessions at a time to bound memory. Each stop pair is its own backtest (`stops` column).

`main` writes the long and the short strategy's trades to `results/trades.parquet` (overwritten on each run).

## Signals and entries

**A signal is a state.** A factory returns a `Strategy` whose `side` is 1 or -1 on every bar where its condition
holds, else 0, known at the bar's close:

| Factory | `side` |
|---|---|
| `rsi_long(freq, level, reverse=False)` | 1 while the `freq` RSI is below `level` (mean reversion); `reverse`: -1 instead (momentum). Name `rsi_long_{freq}_{level}[_reversed]` |
| `rsi_short(freq, level, reverse=False)` | -1 while the `freq` RSI is above `level`; `reverse`: 1 instead |
| `ud_range(freq, range_freq, pivots, level, trend, once, skip_hit, reach)` | Inside the range of `range_freq`'s U/D pivots, on the trend of the `trend` timeframes' U/D flags: 1 at or below the `level` retracement from the top, -1 at or above it from the bottom. Its TP is the gap to the far end (`stops.gap_stops`) |
| `ud_cross(freq, range_freq, pivots, entry, trend, once, reach)` | Continuation: 1 on the bar whose close crosses up through `entry` of the range with the trend up; -1 the mirror |

The RSI is feature engineering's `20_sma_rsi_{freq}` (RSI 14 with simple means), 0–100. On a `FREQ` row, a larger
timeframe's value is its last closed bar's. `rsi_short_60_85_reversed` therefore means **buy** while the 60-min RSI
is above 85.

**`expand`** makes one Strategy per factory x freq x combination of params. **`ensemble(strategies, rule, side,
search_mode)`** turns them into one: each strategy's `side` becomes a column named after it, and `rule` combines
them. With `side=1` or `-1`, the rule sees only that side's signals and its state keeps only that side, so **longs
and shorts run as separate strategies** (`ensemble_function_long` / `_short`): neither blocks or closes the other's
trades. Each trade keeps the signal columns of its signal bar and **`signals`**, a label such as `L110100`: the
position (L / S) and one digit per signal of that side, 1 where it was active.

**`ensemble_function`**: normal mode is strict (1 while every long signal is 1, -1 while every short signal is -1);
search mode is lenient (any signal). 0 when neither or both.

**Which bars enter:**

| | Normal mode (`SEARCH_MODE = False`) | Search mode (`True`) |
|---|---|---|
| A signal | where `side` turns 1 / -1 in a session (from 0 or the other side; a session's first bar counts) | the start of each run of bars with the same `side` and `signals` pattern in a session |
| Signal exit | the opposite signal | the run ends (the pattern changes or `side` goes 0) |
| After a TP / SL exit | the next signal waits for `side` to go back to 0 and turn on again | the next trade waits for a new run |

A signal is skipped when its entry bar has a `NO_ENTRY` column set, when its own bar is outside `VOL_BAND`, or when
its entry would be at or after the flat bar.

## Entries

The signal is known at its bar's close, so the trade enters on the **next `FREQ` bar of the same session**. The TP
and SL levels are always measured from that bar's **open**.

- **`ENTRY = 'market'`**: at the entry bar's first tick (its open), with `SLIPPAGE` in the costs.
- **`ENTRY = 'limit'`**: a limit order at the entry bar's open, sent when the bar starts, reaches the market 300 ms
  later (`stop_search.params.ENTRY_LATENCY_MS`). `entry.get_entry_many` looks up the entry minute's limit-entry
  columns in `1_ohlcv.parquet` by its `start_ind` (0.2 s to load once, about 28 ms per 100,000 entries). For a bar
  above 1 min, the order is valid in its first minute only.
  - Filled at once at the price then, if it is at the open or better (below it for a buy, above it for a sell).
  - Else filled at the open when price comes back to it within the minute.
  - **Skipped** when it never fills, or when the TP or SL level was reached during the latency (`pre_high_1` /
    `pre_low_1`) or while the order waited (the side's `after_high` / `after_low`).
  - The exit walk still starts at the minute's first tick: before the fill no level was reached, so the first hit
    comes after it.
  - **A limit run uses limit orders throughout, so it pays no slippage**: the entry, the TP, the SL as a
    stop-limit at its level, and the signal and flat exits as limits at the next bar's open (see *Costs*).

  Over all minutes 98.6% fill, 88% of them at the open and 12% at a better price (`docs/to_dat.md` has the
  breakdown). With the TP / SL check, a run keeps slightly fewer: 97.8% of the trades in the RSI sweep.

### Setting up limit entries

1. **Step 1 with `ts_ns`.** The latency needs sub-second times: `data/ES_trades_concat.parquet` must have the
   `ts_ns` column (built since 2026-09). An older file: rerun `python -m data_pipeline.raw_data_preprocessing`.
2. **Step 2 writes the entry columns** into `data/processed/1_ohlcv.parquet`: `python -m data_pipeline.to_dat`
   after step 1, and again after changing `ENTRY_LATENCY_MS` in `stop_search/params.py` (300 ms). `tick.dat`, the
   other bar files and the training data do not change, so feature engineering need not rerun.
3. **In `backtesting/config.py`**: `ENTRY = 'limit'` (the default is `'market'`).
4. **Run** `python -m backtesting.backtest`. The entries are read from `1_ohlcv.parquet` next to `tick.dat`
   (`entry.ENTRIES_PATH`), loaded once per process. In code, `find_exits(..., entry='limit')` does the same, and
   `entries=Entries.load(path)` points it at another build.

Check: every trade has `market_entry = 0`, and a run has a few percent fewer trades than with `'market'` (the
entries that never filled, or whose TP / SL was hit before the fill). A bar file without the entry columns stops
the run with "the 1-min bars have no entry columns …: rebuild them with data_pipeline/to_dat.py". `net_units`
equals `pnl` on every trade: no slippage in a limit run.

## Exits

| `result` | Exit | Price | Slippage: market run | Slippage: limit run |
|---|---|---|---|---|
| 1 | Take-profit touched first | the TP level | no (a resting limit) | no |
| -1 | Stop-loss touched first | the SL level | yes (a stop order) | no (a stop-limit at the level) |
| 2 | Signal exit (see above): known at that bar's close | the next bar's first tick | yes | no (a limit at the open) |
| 0 | Flat: still open when the bar holding `FLAT_AT` starts | that bar's first tick (after an early close, the session's last tick) | yes | no (a limit at the open) |

Every position is flat by the session's end. The exit walk uses `tick.dat` through `stop_search`, so a TP and an SL
inside the same bar are resolved on the ticks.

## Costs (`add_costs`)

- `gross_usd` = `pnl` x `POINT_VALUE`.
- `net_units` = `pnl` − `SLIPPAGE` x (market fills). A market run (`market_entry = 1`): 1 for the entry, plus 1
  unless the exit is a TP. A limit run (`market_entry = 0`): 0, limit orders throughout.
- `net_usd` = `net_units` x `POINT_VALUE` − 2 x `COMMISSION`.

With the defaults: a market run's stop-loss trade costs 2 units ($25) plus $3.80, its take-profit trade 1 unit plus
$3.80; every limit-run trade costs only the $3.80.

## The trades table

One row per trade, every strategy and stop pair together:

| Column | Meaning |
|---|---|
| `strategy`, and its params | e.g. `ensemble_function_long` (an ensemble has no params columns) |
| `stops`, `tp`, `sl` | the run label (e.g. `40/24`) and the distances in units |
| `row`, `signal_ts`, `date` | the signal bar's row in the `FREQ` frame, its time, the session |
| `side` | 1 long, -1 short |
| `LABEL_FEATURES`, the signal columns, `signals` | from the signal bar (known when the order goes in) |
| `entry_ts`, `year`, `month`, `session`, `rth`, `hour`, `news_{FREQ}` | the entry bar and its labels |
| `entry_ind`, `entry_px`, `market_entry` | `tick.dat` row, entry price (the fill for a limit), 1 market / 0 limit |
| `exit_row`, `exit_ts`, `exit_px`, `result` | the exit bar, time, price and kind (above) |
| `pnl`, `gross_usd`, `net_units`, `net_usd` | gross in units, gross $, net in units, net $ |

## Analysis

`python -m backtesting.analysis` reads the trades table and:

- prints the **leaderboard** (`BacktestStats.grid` by strategy and stops: trades, win rate, gross, costs, net, max
  drawdown, Sharpe);
- slices every run by label (hour, news, volatility bucket, year, month, side, each crossed with side, the
  `signals` pattern, and every label at once), with the t-stat of the net per trade and the net before / after
  `SPLIT`;
- runs a **chance check**: how many slices are positive on both sides of `SPLIT` against what chance alone gives;
- charts hours, labels and stability in `results/analysis/`, and the final run (`FINAL`, else the best net) in
  `results/final/`.

For your own queries: `load_trades()`, `slices(trades)`, `pooled(trades)`, and
`BacktestStats(trades.filter(...)).summary() / .daily() / .by_period('month') / .by_label('hour') / .by_side()`.
`stats.summary()`'s drawdown is trade by trade on the net, from a peak starting at 0.

## Judging a result

Thousands of runs are tried, so the best ones look good partly by luck. These checks were used for the RSI study
(`summary/rsi.md`):

- **Drift check**: the run against random entries on the same side, with the same stops and entry model, in the
  **same month and hour of day** as its trades. Don't match on the same day and hour: a random entry can then land
  before the signal in its hour and catch the move that caused it, which flatters mean-reversion longs and punishes
  momentum shorts. Random longs gain from ES's rise over 2020–2026, so a long must beat them, not just be positive.
- **Pick before a date, test after it**: choose on 2020–2025 and look at 2026 once.
- **Calmar** (net $ per year ÷ max drawdown) to favour smooth curves, with a minimum number of trades.

The drift check script is not in the repo yet (see `summary/rsi.md`); `walk_forward.py` covers picking from the past.

## Research tools

For many signals at once, outside `config.py`'s single run. Each writes to `results/`.

```bash
python -m backtesting.sweep ud_range                      # 1-min execution near a timeframe's last-two-pivot range (~15 min)
python -m backtesting.sweep ud_near --freq 5,10,15,30,60  # near any of a timeframe's last 5 pivots (~20 min)
python -m backtesting.sweep rsi                           # the RSI levels, normal and reversed (~7 min)
python -m backtesting.pairs ud_range --breakdown          # best long + short pair per condition, plot and statistics
python -m backtesting.pairs limit --positive-legs         # the RSI study's pairs: only legs profitable on their own
python -m backtesting.pairs limit --positive-legs --portfolio   # + the best-Calmar portfolio of pairs, one panel per pair
python -m backtesting.pairs ud_range --look ud            # the UD plots' look (summary/ud.md)
python -m backtesting.walk_forward --trades 'results/sweep/ud_range/trades_*.parquet' --train 5 --test 1 --opposite
```

- **`sweep.py`**: every signal of a preset as its own long and short strategy, over the fixed TP / SL grid (16..100
  units), the volatility grid (1..6 x `20_std_{freq}`) or both (`--stops`), limit entries by default, with
  `config.py`'s news and volatility-band rules. One table of trades per trade freq in `results/sweep/{preset}/`, plus a
  summary per signal x stops. `ud_range` trades also keep `near`: every timeframe whose range end was close at the
  signal bar (e.g. `5,15`).
- **`pairs.py`**: a pair is one condition (same preset, bar size, level or range, distance), a long run and a short run
  of it, differing only in side and stops. For every condition, every long run x every short run, summed by
  session and ranked by Calmar; mirror pairs (a short whose stops are the long's swapped, the same trade both ways)
  are left out. Writes `pairs.parquet` and `acc_pnl_pairs.png` and prints each top pair's and leg's statistics;
  `--breakdown` adds net per trade by session, regular hours and `near`; `--portfolio` adds, from the best pair, the
  pair of another condition that raises the Calmar most while it rises by 0.02 (`portfolio.parquet`,
  `acc_pnl_portfolio.png`: the portfolio and each pair in one panel; `acc_pnl_portfolio_pairs.png`: a panel per
  pair with its long and short legs). All in sample.
- **`walk_forward.py`**: out of sample. Each side picks its best run (net or Calmar) over the last `--train` hours,
  days or months, only from trades already closed, and trades it (or with `--opposite` its mirror) for the next
  `--test`; only the test periods are recorded.

## No look-ahead

- Signals use data up to their bar's close; the entry is the next bar.
- On a `FREQ` row, larger timeframes show their last closed bar, and smaller ones are as of the bar's close.
- `vol_ok` uses the previous sessions only.
- The limit-entry columns describe the entry minute itself. They decide only whether and at what price the order
  fills, never whether to trade.
- `analysis.feature_cuts` buckets `LABEL_FEATURES` with quantiles over the whole history: fine to describe trades,
  not for a trading rule.

## ML (parked)

`python -m backtesting.ml` builds features and an outcome table (every bar entered long and short with every stop
pair, exits by TP / SL / flat only). `ml_ensemble` (a replacement `ENSEMBLE`) and `ml_stops` (a replacement `STOP`)
train LightGBM walk-forward, each test year on the years before it, and write predictions to `results/ml/`. Set
aside for now; not part of the current research.

## Tests

`tests/test_backtest.py` (find_exits against a tick-by-tick scan for market and limit entries, one_at_a_time,
costs, the grid, search mode, the ensemble, the strategies, `vol_ok`, `vol_grid`, `get_entry_many`),
`tests/test_entry.py` (the limit-entry columns) and `tests/test_stats.py`. `pytest` runs them without the real data;
the test wrappers trade 1-min bars whatever `config.FREQ` says.

## Known limitations

- **Limit fills are optimistic.** A trade at the open counts as a fill for both a buy and a sell, but only one side
  is marketable at a time. About 10% of fills may not happen, and they tend to be winners. Small-TP strategies are
  the most exposed. Fix: keep the raw aggressor `side` in step 1 and use a spread-and-queue rule in `to_dat.py`.
- **The TP fills on touch.** A resting limit at the TP counts as filled when price reaches it, with the same queue
  question.
- **A limit run's exits always fill at their price.** The SL as a stop-limit at its level can miss in a fast market
  (price jumps past it) and leave the position open; the signal and flat exits, as limits at the next bar's open, can
  miss too. The backtest assumes they fill; a gap or chase model is not built. In a market run the SL fills at its
  level plus 1 unit, which can also be too kind in a fast market.
- **Longs and shorts run separately**, so a long and a short can be open at once. In one account they net; the PnL
  is the same, since it is linear.
- Commission and slippage are fixed assumptions ($1.90 per side; 1 unit per market fill in a market run, none in a
  limit run).
