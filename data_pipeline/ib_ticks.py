"""
One session of IB-recorded ES trades from MongoDB, as a raw table raw_data_preprocessing.to_ticks accepts.

Assumed documents, one per trade (field names are set in FIELDS below):
    {"time": 1727208000, "price": 5100.25, "size": 3, "symbol": "ES"}

LocalCollection stands in for MongoDB with documents from a JSON-lines file, e.g. the fake recorder data
from fake_ib_data.py (`python -m data_pipeline.daily_update append --file ...`).
"""
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

NY = "America/New_York"

# Where the IB recorder writes; environment variables override the defaults (handy in a crontab)
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "ib")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "ES_trades")

# Document field names, and how `time` is stored: "s" / "ms" for epoch numbers, "datetime" for BSON dates
FIELDS = {"time": "time", "price": "price", "size": "size", "symbol": "symbol"}
TIME_UNIT = "s"
SYMBOL = "ES"


def load_holidays(path=Path(__file__).resolve().parents[1] / "params" / "holidays.yaml"):
    """Exchange holidays from params/holidays.yaml: a list of dates, or a mapping of date -> name."""
    with open(path) as f:
        data = yaml.safe_load(f) or []
    return {datetime.date.fromisoformat(str(d)) for d in data}


# A holiday with no ticks is skipped by append; any other weekday with no ticks stops it (see the yaml)
HOLIDAYS = load_holidays()


class LocalCollection:
    """
    A stand-in for the MongoDB collection: documents held in memory, answering the one query fetch_session
    makes (a time range and a symbol, sorted by time, then insertion order). Documents keep their order,
    which plays the part of MongoDB's insertion order (_id).
    """

    def __init__(self, docs):
        self.docs = [dict(doc, _id=i) for i, doc in enumerate(docs)]

    @classmethod
    def from_file(cls, path):
        """Documents from a JSON-lines file, one per line (fake_ib_data.py's output, or a mongoexport)."""
        with open(path) as f:
            return cls(_from_json(json.loads(line)) for line in f if line.strip())

    def find(self, query, projection):
        def match(doc):
            for field, cond in query.items():
                value = _utc(doc.get(field))
                if isinstance(cond, dict):
                    if value is None or not _utc(cond["$gte"]) <= value < _utc(cond["$lt"]):
                        return False
                elif value != cond:
                    return False
            return True

        return _Cursor({k: doc[k] for k in ("_id", *projection)} for doc in self.docs if match(doc))


class _Cursor(list):
    def sort(self, keys):
        return _Cursor(sorted(self, key=lambda doc: tuple(doc[k] for k, _ in keys)))


def _utc(value):
    """Datetimes compared as naive UTC, as MongoDB does (it stores them without a time zone)."""
    if isinstance(value, datetime.datetime) and value.tzinfo is not None:
        return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return value


def _from_json(doc):
    """mongoexport writes dates as {"$date": "2026-09-25T13:30:00Z"}; turn them back into datetimes."""
    return {k: _utc(datetime.datetime.fromisoformat(v["$date"].replace("Z", "+00:00"))) if isinstance(v, dict) and "$date" in v else v
            for k, v in doc.items()}


def connect(uri=None, db=None, collection=None):
    """The MongoDB collection holding the recorded trades."""
    from pymongo import MongoClient  # only needed when reading from MongoDB
    return MongoClient(uri or MONGO_URI)[db or MONGO_DB][collection or MONGO_COLLECTION]


def session_window(session_date):
    """UTC [start, end) of a session: New York 18:00 the day before to 17:00 on session_date."""
    day = pd.Timestamp(session_date)
    start = (day - pd.Timedelta(hours=6)).tz_localize(NY).tz_convert("UTC")
    end = (day + pd.Timedelta(hours=17)).tz_localize(NY).tz_convert("UTC")
    return start, end


def last_closed_session(now=None):
    """
    The newest session that has closed (17:00 New York) by `now` (tz-aware; default the current time): today
    once it is past 17:00 on a weekday, otherwise the weekday before. Saturday and Sunday have no session of
    their own (Sunday evening opens Monday's).
    """
    now = pd.Timestamp.now(tz=NY) if now is None else pd.Timestamp(now).tz_convert(NY)
    day = now.normalize().tz_localize(None)
    if now.hour < 17:
        day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day.date()


def session_dates(after, until):
    """Weekdays after `after` up to and including `until`: the sessions that may have traded in between."""
    days = pd.date_range(pd.Timestamp(after) + pd.Timedelta(days=1), pd.Timestamp(until), freq="D")
    return [d.date() for d in days if d.weekday() < 5]


def fetch_session(collection, session_date, fields=FIELDS, symbol=SYMBOL, time_unit=TIME_UNIT):
    """
    One session's trades as a raw table (ts_event UTC, price, size), in recorded order (time, then insertion
    order), ready for raw_data_preprocessing.to_ticks. Empty if MongoDB holds no trades for the session.
    Prices are rounded to the 0.25 grid (float noise); a price clearly off it raises ValueError.
    The index {symbol: 1, time: 1, _id: 1} serves this query and its sort without an in-memory sort.
    """
    start, end = session_window(session_date)
    if time_unit == "datetime":
        lo, hi = start.to_pydatetime(), end.to_pydatetime()
    else:
        scale = {"s": 1, "ms": 1000}[time_unit]
        lo, hi = int(start.timestamp()) * scale, int(end.timestamp()) * scale
    query = {fields["time"]: {"$gte": lo, "$lt": hi}}
    if symbol:
        query[fields["symbol"]] = symbol
    projection = {fields[k]: 1 for k in ("time", "price", "size")}
    docs = list(collection.find(query, projection).sort([(fields["time"], 1), ("_id", 1)]))

    times = [d[fields["time"]] for d in docs]
    if time_unit == "datetime":
        ts = pd.to_datetime(times, utc=True)
    else:
        ts = pd.to_datetime(np.asarray(times, dtype=np.int64), unit=time_unit, utc=True)
    price = np.array([float(d[fields["price"]]) for d in docs], dtype=np.float64)
    ticks = np.rint(price * 4)
    off = np.abs(price * 4 - ticks) > 1e-6
    if off.any():
        raise ValueError(f"session {session_date}: {off.sum()} prices off the 0.25 grid, e.g. {price[off][0]}")
    return pa.table({
        "ts_event": pa.array(ts, pa.timestamp("ns", tz="UTC")),
        "price": pa.array(ticks / 4, pa.float64()),
        "size": pa.array([int(d[fields["size"]]) for d in docs], pa.int64()),
    })
