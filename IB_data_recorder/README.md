# IB Data Recorder

A script to record historical bar and tick data from Interactive Brokers.

## Prerequisites

1. You need an active Interactive Brokers account.
2. IB Gateway (or TWS) must be running. If you're using IBC (IB Controller) to manage Gateway, make sure it's started and logged in.
3. By default, IB Gateway live trading uses port `4001` and paper trading uses port `4002`.
   - The script is set to default to port `4001`.
   - Add `127.0.0.1` to the **"Trusted IPs"** if not already there.

## Usage

### Record Tick Data (e.g. ES futures)

For tick data, you must provide `--tick` and the `--num-ticks` to retrieve. TWS API only allows up to 1000 ticks per request.

```bash
python ib_data_recorder.py \
    --symbol ES \
    --sec-type FUT \
    --exchange CME \
    --currency USD \
    --tick \
    --num-ticks 1000 \
    --what-to-show TRADES \
    --port 4001
```

### Record Bar Data

```bash
python ib_data_recorder.py \
    --symbol AAPL \
    --sec-type STK \
    --exchange SMART \
    --currency USD \
    --duration "1 Y" \
    --bar-size "1 day" \
    --what-to-show TRADES \
    --port 4001
```

### Options

* `--symbol`: The ticker symbol (e.g. AAPL, ES)
* `--sec-type`: Security type (STK, FUT, CASH, OPT, etc)
* `--exchange`: Exchange (SMART, CME, IDEALPRO, etc)
* `--currency`: Currency (USD, EUR, etc)
* `--tick`: Request tick data instead of bar data
* `--num-ticks`: Number of ticks (max 1000)
* `--duration`: Duration for bar data (e.g. "1 Y", "1 M", "1 W", "1 D", "3600 S")
* `--bar-size`: Bar size (e.g. "1 day", "1 min", "5 mins", "1 hour")
* `--what-to-show`: What data to show (TRADES, MIDPOINT, BID_ASK, BID, ASK)
* `--rth`: 1 for Regular Trading Hours, 0 for all hours
* `--end-datetime`: Format 'YYYYMMDD HH:MM:SS' (empty for current time)
* `--host`: TWS or Gateway IP address (default 127.0.0.1)
* `--port`: TWS or Gateway port (default 4002)
* `--client-id`: Client ID for API connection
* `--output`: Custom output path
* `--parquet`: Save as Parquet instead of CSV
