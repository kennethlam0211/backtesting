from .params import CHILD, DAT_COLS, DAT_PATH, FREQS, PARQUET_FREQS

__all__ = ['StopSearch', 'CHILD', 'DAT_COLS', 'DAT_PATH', 'FREQS', 'PARQUET_FREQS']
_LAZY = ('StopSearch',)


def __getattr__(name):
    # The numba code loads on first use, so `from stop_search.params import ...` (data_pipeline/to_dat.py) stays light
    if name in _LAZY:
        from . import stop_search
        return getattr(stop_search, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
