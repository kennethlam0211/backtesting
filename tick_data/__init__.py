from .params import CHILD, DAT_COLS, DAT_PATH, FREQS, PARQUET_FREQS

_LAZY = ('PRICE_COL', 'TickData', 'first_hit', 'first_hit_many', 'load_dat')


def __getattr__(name):
    # The numba code loads on first use, so `from tick_data.params import ...` (pipeline/to_dat.py) stays light
    if name in _LAZY:
        from . import tick_data
        return getattr(tick_data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
