import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / 'training' / 'log'

# Track created loggers by base name to prevent duplicates across calls
_created_loggers = {}


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Get a logger that writes to training/log/YYYYMMDD/<name>.log

    Logger name includes HMS to identify the run in output.

    Usage:
        from app_logger import get_logger
        logger = get_logger('env_v2')
        logger.debug(some_dict)       # file only
        logger.info("exam passed")    # file + console
    """
    import params
    mode = getattr(params, 'TRAINING_MODE', '')
    base_key = f'{name}_{mode}'

    # Return existing logger if already created for this base name
    if base_key in _created_loggers:
        return _created_loggers[base_key]

    now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S')
    logger_key = f'{name}_{mode}_{date_str}_{time_str}'
    logger = logging.getLogger(logger_key)

    if logger.handlers:
        _created_loggers[base_key] = logger
        return logger

    logger.setLevel(level)
    logger.propagate = False

    day_dir = LOG_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    # File handler — all levels
    fh = logging.FileHandler(day_dir / f'{name}.log', mode='a', encoding='utf-8')
    fh.setLevel(level)
    fmt = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — INFO+ only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    _created_loggers[base_key] = logger
    return logger
