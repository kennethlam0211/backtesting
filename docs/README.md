# Docs

ES futures research: raw trades → ticks and bars → features → backtests. Run everything from the repo root with
`python -m ...`; paths are set in `params/params.py`.

1. `data_pipeline.raw_data_preprocessing`: `raw_data/` → `data/ES_trades_concat.parquet` (one tick file)
2. `data_pipeline.to_dat`: → `data/processed/tick.dat` and `{freq}_ohlcv.parquet` (the 1-min file with the
   limit-entry columns)
3. `data_pipeline.feature_engineering`: the bar files → `data/processed/training_data.parquet`
4. `backtesting.backtest`: the training data, `tick.dat` and the 1-min entry columns → `results/trades.parquet`
5. `backtesting.analysis`: the trades → leaderboard, label analysis, charts in `results/`

| Doc | Covers |
|---|---|
| [`data_pipeline/README.md`](../data_pipeline/README.md) | How to run the pipeline: `daily_update init` / `append` (IB sessions from MongoDB, cron), each step's command, inputs and outputs, step 3's features |
| [`data_pipeline.md`](data_pipeline.md) | Step 1 in detail: the clock (New York + 6h), roll days, merging, news flags, checks |
| [`to_dat.md`](to_dat.md) | Step 2 in detail: the `tick.dat` layout, how bars are built, how the stop search reads them, the limit-entry columns |
| [`backtesting.md`](backtesting.md) | The backtester: settings, signals, entries (market / limit), exits, costs, the trades table, analysis, how to judge a result |
| [`../summary/rsi.md`](../summary/rsi.md) | Research results: RSI strategies with limit entries, single runs, long + short pairs, the best 5 portfolio |
| [`../summary/ud.md`](../summary/ud.md) | Research results: UD pivot pairs, the 1-min range version (kept) and the near-pivot version |

Prices are in ticks (points x 4) everywhere: 1 unit = 0.25 point = $12.50 for ES.
