from .params import (
    CHILD,
    DAT_COLS,
    DAT_DTYPE,
    DAT_PATH,
    FREQS,
    INDEX_COLS,
    PARQUET_FREQS,
    PRICE_COLS,
    ROW_BYTES,
)

__all__ = ['StopSearch', 'CHILD', 'DAT_COLS', 'DAT_DTYPE', 'DAT_PATH', 'FREQS', 'INDEX_COLS', 'PARQUET_FREQS',
           'PRICE_COLS', 'ROW_BYTES']
_LAZY = ('StopSearch',)


def __getattr__(name):
    # The numba code loads on first use, so `from stop_search.params import ...` (data_pipeline/to_dat.py) stays light
    if name in _LAZY:
        from . import stop_search
        return getattr(stop_search, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
