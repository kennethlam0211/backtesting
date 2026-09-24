import argparse
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.traceback import install

install()
console = Console()

# Add IB_data_recorder/libs to path to import ibapi
PROJECT_PATH = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_PATH / 'libs'))

try:
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper
except ImportError as e:
    console.print(f"[red]Failed to import ibapi: {e}[/red]")
    console.print(f"Make sure ibapi is available at {PROJECT_PATH / 'libs'}")
    sys.exit(1)


class IBDataRecorder(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.req_id = 1
        self.data_ready = threading.Event()
        self.error_event = threading.Event()

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.req_id = orderId
        console.print(f"[green]Connected to TWS/Gateway. Next valid ID: {orderId}[/green]")

    def historicalData(self, reqId, bar):
        self.data.append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "barCount": bar.barCount,
            "wap": bar.wap
        })

    def historicalDataEnd(self, reqId, start, end):
        console.print(f"[green]Finished receiving historical data from {start} to {end}[/green]")
        self.data_ready.set()

    def historicalTicks(self, reqId, ticks, done):
        for tick in ticks:
            self.data.append({
                "time": tick.time,
                "price": tick.price,
                "size": tick.size,
                "type": "MIDPOINT"
            })
        if done:
            console.print("[green]Finished receiving historical ticks (MIDPOINT)[/green]")
            self.data_ready.set()

    def historicalTicksBidAsk(self, reqId, ticks, done):
        for tick in ticks:
            self.data.append({
                "time": tick.time,
                "priceBid": tick.priceBid,
                "priceAsk": tick.priceAsk,
                "sizeBid": tick.sizeBid,
                "sizeAsk": tick.sizeAsk,
                "type": "BID_ASK"
            })
        if done:
            console.print("[green]Finished receiving historical ticks (BID/ASK)[/green]")
            self.data_ready.set()

    def historicalTicksLast(self, reqId, ticks, done):
        for tick in ticks:
            self.data.append({
                "time": tick.time,
                "price": tick.price,
                "size": tick.size,
                "exchange": tick.exchange,
                "specialConditions": tick.specialConditions,
                "type": "TRADES"
            })
        if done:
            console.print("[green]Finished receiving historical ticks (TRADES)[/green]")
            self.data_ready.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        super().error(reqId, errorCode, errorString, advancedOrderRejectJson)
        # 2104, 2106, 2158 are informational messages about connection
        if errorCode not in [2104, 2106, 2158]:
            console.print(f"[red]Error {errorCode}: {errorString}[/red]")
            if errorCode in [162, 200]:  # Historical data error / No security definition
                self.error_event.set()
                self.data_ready.set()  # Unblock if there's an error


def create_contract(symbol: str, sec_type: str, exchange: str, currency: str) -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.exchange = exchange
    contract.currency = currency
    return contract


def fetch_historical_data(
    app: IBDataRecorder,
    contract: Contract,
    end_datetime: str = "",
    duration: str = "1 Y",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    format_date: int = 1,
    is_tick: bool = False,
    num_ticks: int = 1000
) -> pd.DataFrame:

    app.data = []
    app.data_ready.clear()
    app.error_event.clear()

    if is_tick:
        console.print(f"Requesting TICK data for {contract.symbol} ({num_ticks} ticks)...")
        app.reqHistoricalTicks(
            app.req_id,
            contract,
            startDateTime="",
            endDateTime=end_datetime,
            numberOfTicks=num_ticks,
            whatToShow=what_to_show,
            useRth=use_rth,
            ignoreSize=False,
            miscOptions=[]
        )
    else:
        console.print(f"Requesting BAR data for {contract.symbol} ({duration}, {bar_size})...")
        app.reqHistoricalData(
            app.req_id,
            contract,
            end_datetime,
            duration,
            bar_size,
            what_to_show,
            use_rth,
            format_date,
            False,
            []
        )

    app.req_id += 1

    # Wait for the data to be received
    app.data_ready.wait(timeout=60)

    if app.error_event.is_set():
        console.print("[red]Failed to fetch data due to an error.[/red]")
        return pd.DataFrame()

    if not app.data:
        console.print("[yellow]No data received or request timed out.[/yellow]")
        return pd.DataFrame()

    df = pd.DataFrame(app.data)
    return df


def main():
    parser = argparse.ArgumentParser(description="Record historical data from Interactive Brokers.")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--sec-type", type=str, default="STK", help="Security type (e.g., STK, FUT, CASH)")
    parser.add_argument("--exchange", type=str, default="SMART", help="Exchange (e.g., SMART, IDEALPRO)")
    parser.add_argument("--currency", type=str, default="USD", help="Currency (e.g., USD, EUR)")
    parser.add_argument("--duration", type=str, default="1 Y", help="Duration string (e.g., '1 Y', '1 M', '1 W', '1 D')")
    parser.add_argument("--bar-size", type=str, default="1 day", help="Bar size (e.g., '1 day', '1 min', '5 mins')")
    parser.add_argument("--what-to-show", type=str, default="TRADES", help="What to show (e.g., TRADES, MIDPOINT, BID, ASK)")
    parser.add_argument("--rth", type=int, default=1, choices=[0, 1], help="Use regular trading hours only (1) or outside RTH (0)")
    parser.add_argument("--end-datetime", type=str, default="", help="End datetime in format 'YYYYMMDD HH:MM:SS' (empty for current time)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="TWS or Gateway IP address")
    parser.add_argument("--port", type=int, default=4001, help="TWS or Gateway port (TWS: 7496/7497, Gateway: 4001/4002)")
    parser.add_argument("--client-id", type=int, default=999, help="Client ID for API connection")
    parser.add_argument("--output", type=str, default="", help="Output CSV path (e.g., data/aapl.csv)")
    parser.add_argument("--parquet", action="store_true", help="Save as Parquet format instead of CSV")
    parser.add_argument("--tick", action="store_true", help="Request tick data instead of bar data")
    parser.add_argument("--num-ticks", type=int, default=1000, help="Number of ticks to request if --tick is used")

    args = parser.parse_args()

    app = IBDataRecorder()

    app.connect(args.host, args.port, clientId=args.client_id)

    # Start the IB API message loop in a background thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Give it a moment to connect and receive the next valid ID
    time.sleep(1)

    if not app.isConnected():
        console.print(f"[red]Failed to connect to TWS/Gateway at {args.host}:{args.port}[/red]")
        sys.exit(1)

    contract = create_contract(args.symbol, args.sec_type, args.exchange, args.currency)

    try:
        df = fetch_historical_data(
            app,
            contract,
            end_datetime=args.end_datetime,
            duration=args.duration,
            bar_size=args.bar_size,
            what_to_show=args.what_to_show,
            use_rth=args.rth,
            is_tick=args.tick,
            num_ticks=args.num_ticks
        )

        if not df.empty:
            console.print(df.head())
            console.print(df.tail())

            # Determine default output path if not provided
            if not args.output:
                safe_symbol = args.symbol.replace(" ", "_")
                ext = "parquet" if args.parquet else "csv"
                time_str = "latest" if not args.end_datetime else args.end_datetime.replace(" ", "_").replace(":", "")
                data_type = "tick" if args.tick else args.bar_size.replace(' ', '')
                out_path = Path("data") / f"{safe_symbol}_{data_type}_{time_str}.{ext}"
            else:
                out_path = Path(args.output)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.suffix == '.parquet' or args.parquet:
                df.to_parquet(out_path, index=False)
            else:
                df.to_csv(out_path, index=False)

            console.print(f"[bold green]Saved {len(df)} rows to {out_path}[/bold green]")
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user[/yellow]")
    finally:
        app.disconnect()
        console.print("Disconnected.")

if __name__ == "__main__":
    main()
