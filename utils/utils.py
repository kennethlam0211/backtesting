import os
import sys
import yaml
import datetime
import json
import time
import copy
import collections.abc
import inspect
import traceback
from functools import wraps, reduce
import operator
import numpy as np
import psutil
import pandas as pd
from collections import OrderedDict
from pymongo import MongoClient

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import MGDB_CONFIG, TRADING_ENV, TRADING_ENV_CONFIG

TRADING_ENV_CONFIG = TRADING_ENV_CONFIG[TRADING_ENV.upper()]


# ── MongoDB ─────────────────────────────────────────────────────────────

def get_mgdb_client(where='local'):
    """Get MongoDB client.
    where: 'local' (localhost) or 'server' (192.168.50.238)
    """
    return MongoClient(MGDB_CONFIG[where])


def get_db(db_name, where='local'):
    """Get MongoDB database."""
    return get_mgdb_client(where)[db_name]


def get_col(db_name, col_name, where='local', indices=None, tag='tag', unique=False, trading_env=None):
    """Get MongoDB collection."""
    try:
        myclient = get_mgdb_client(where=where)
        if not trading_env:
            trading_env = TRADING_ENV
        if trading_env == 'STAGE' and db_name not in ['TRADING_MEMORY', 'HSI_HIST_DATA_IB', 'resampled_db', 'STAGE_LOG_DB', 'ta_db']:
            db_name = f'{TRADING_ENV}_{db_name}'
        mycol = myclient[db_name][col_name]
        if indices:
            indices = [(item, 1) for item in indices]
            index_list = mycol.index_information()
            if tag not in index_list:
                mycol.create_index(indices, unique=unique, name=tag)
        return mycol
    except Exception as e:
        get_exception_msg()


def update_db(col, query, update_key, upsert=False, hint="_id_", many=False):
    try:
        func = col.update_many if many else col.update_one
        func(query, {"$set": update_key}, upsert=upsert, hint=hint)
    except Exception as e:
        get_exception_msg()


# ── Dict utilities ──────────────────────────────────────────────────────

class mydict(dict):
    def __missing__(self, key, value=None):
        value = self[key] = type(self)()
        return value

    def __add__(self, x):
        if not self:
            return x
        raise ValueError

    def __sub__(self, x):
        if not self:
            return -x if isinstance(x, (int, float)) else x
        raise ValueError


class LimitedDictWithDefault(OrderedDict):
    def __init__(self, max_len=None, default=None):
        super().__init__()
        self.max_len = max_len
        self.default = default

    def __setitem__(self, key, value):
        if len(self) >= self.max_len:
            self.popitem(last=False)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if key not in self:
            self[key] = copy.copy(self.default)
        return super().__getitem__(key)


class mydictWrapper:
    def __init__(self, kwargs={}):
        self.update(kwargs)

    def update(self, kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            default_value = mydict()
            setattr(self, key, default_value)
            return default_value

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def pop(self, key, default=None):
        if hasattr(self, key):
            value = getattr(self, key)
            delattr(self, key)
            return value
        return None

    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default

    def __str__(self):
        return json.dumps(vars(self), default=dump_hook)

    def __call__(self, deep=True):
        return copy.deepcopy(vars(self)) if deep else vars(self)

    def keys(self):
        return self.__dict__.keys()

    def __len__(self):
        return len(self.keys())

    def __bool__(self):
        return len(self.keys()) > 0

    @property
    def empty(self):
        return not list(self.keys())


def deep_get(dataDict, keys):
    if isinstance(keys, str):
        keys = keys.split(".")
    return reduce(operator.getitem, keys, dataDict)


def deep_set(dataDict, keys, value, mode=None):
    if isinstance(keys, str):
        keys = keys.split(".")
    parent = deep_get(dataDict, keys[:-1])
    try:
        if mode == '+=':
            parent[keys[-1]] += value
        elif mode == '-=':
            parent[keys[-1]] -= value
        elif mode == 'update':
            if isinstance(value, dict):
                parent[keys[-1]] = deep_update(parent.get(keys[-1], {}), value)
            else:
                parent[keys[-1]] = value
        else:
            parent[keys[-1]] = value
    except KeyError:
        parent[keys[-1]] = value
    except Exception:
        get_exception_msg()


def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            source[key] = deep_update(source.get(key, {}), value)
        else:
            source[key] = overrides[key]
    return source


def deep_get_key(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                yield v
            else:
                yield from deep_get_key(v, key)
    elif isinstance(d, list):
        for v in d:
            yield from deep_get_key(v, key)


def deep_pop(dataDict, keys):
    if isinstance(keys, str):
        keys = keys.split(".")
    if len(keys) == 1:
        dataDict.pop(keys[-1], None)
    else:
        parent_key = deep_get(dataDict, keys[:-1])
        parent_key.pop(keys[-1], None)


# ── Datetime utilities ──────────────────────────────────────────────────

def dt_to_str(dt, format="%Y%m%d %H:%M:%S"):
    return datetime.datetime.strftime(dt, format)


def str_to_dt(dt, format="%Y%m%d %H%M%S"):
    return datetime.datetime.strptime(dt, format)


def datetime_to_unix(dt):
    return time.mktime(dt.timetuple())


def unix_to_datetime(dt):
    return datetime.datetime.fromtimestamp(int(dt))


def get_time_str(str_fmt="%H%M%S"):
    return datetime.datetime.now().time().strftime(str_fmt)


def get_log_col_name(dt=None):
    if not dt:
        dt = datetime.datetime.now()
    if dt.hour >= 8:
        return dt.strftime("%Y%m%d")
    return (dt - datetime.timedelta(days=1)).strftime("%Y%m%d")


# ── YAML ────────────────────────────────────────────────────────────────

def yaml_loader(filepath):
    with open(filepath, "r") as f:
        op = mydict()
        res = yaml.safe_load(f)
        if res:
            op.update(res)
        return op


def yaml_dumper(filepath, data, error_handle=False):
    try:
        data = copy.deepcopy(data)
        with open(filepath, "w") as f:
            if isinstance(data, mydict):
                data = dict(data)
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception:
        if not error_handle:
            yaml_dumper(filepath, data, error_handle=True)
        get_exception_msg()


# ── JSON ────────────────────────────────────────────────────────────────

def dump_hook(obj, str_fmt="%Y%m%d %H%M%S"):
    try:
        if isinstance(obj, datetime.datetime):
            return obj.strftime(str_fmt)
        elif isinstance(obj, datetime.time):
            return obj.strftime("%H%M%S")
        elif callable(obj):
            return obj.__name__
        elif isinstance(obj, dict):
            return json.dumps(obj, default=dump_hook)
        elif isinstance(obj, object):
            try:
                return repr(obj)
            except:
                return obj.__class__.__name__
        else:
            return str(obj)
    except Exception:
        return None


# ── Process utilities ───────────────────────────────────────────────────

def mytermination(pid):
    try:
        if isinstance(pid, str):
            pid = int(pid)
        if psutil.Process(pid):
            psutil.Process(pid).terminate()
    except:
        pass


def check_pid(pid_):
    try:
        os.kill(pid_, 0)
    except OSError:
        return False
    return True


# ── Decorators ──────────────────────────────────────────────────────────

def timeit(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        t1 = time.monotonic()
        res = func(self, *args, **kwargs)
        print(func.__name__, ' time taken ', time.monotonic() - t1)
        return res
    return _wrapper


def check_is_running(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        func_name = func.__name__
        if not self.is_running_dict.get(func_name):
            self.is_running_dict.update({func_name: True})
            res = func(self, *args, **kwargs)
            self.is_running_dict.update({func_name: False})
            return res
    return _wrapper


# ── Error handling ──────────────────────────────────────────────────────

def get_exception_msg():
    inspect_ = inspect.stack(context=1)
    function_names = []
    try:
        for i in reversed(range(len(inspect_))):
            function_names.extend(inspect_[i].code_context)
    except:
        pass
    res = {'ts': dt_to_str(datetime.datetime.now()), "called_by": function_names}
    print(res)
    try:
        traceback.print_exc()
    except:
        pass
    res = json.dumps(res, default=dump_hook).replace('\\\\', '\\').replace('\\n', ' ').replace("  ", " ")
    return res


# ── Misc ────────────────────────────────────────────────────────────────

def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if x not in seen and not seen.add(x)]


def same_sign(x, y):
    return (x * y) > 0


def format_val(val):
    return '{:.0f}'.format(val)


def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def flatten(nested_list, iterable_list='iterable'):
    from collections.abc import Iterable
    target_object = {'iterable': Iterable, 'list': list}
    flatten_list = []
    def get_the_nested_item(x):
        [flatten_list.extend([i]) if not isinstance(i, target_object.get(iterable_list)) else get_the_nested_item(i) for i in x]
    get_the_nested_item(nested_list)
    return flatten_list


def is_holidays(d):
    if d.weekday() == 5:  # Saturday
        return d.time() >= datetime.time(6)
    elif d.weekday() == 6:  # Sunday
        return d.time() <= datetime.time(21)
    return False


def resampler_(df):
    if not df.empty:
        high_ind = df['high'].idxmax()
        low_ind = df['low'].idxmin()
        return pd.Series({
            'ts': df['ts'].iloc[0],
            'open': df['open'].iloc[0],
            'high': df.at[high_ind, 'high'],
            'low': df.at[low_ind, 'low'],
            'close': df['close'].iloc[-1],
            'volume': df['volume'].sum(),
            'avg_px': ((df['avg_px'] * df['volume']).sum()) / (df['volume'].sum()),
            'hl': 1 if high_ind < low_ind else -1,
        })
