import os
import sys
import logging
import datetime
from copy import copy

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.mgdb_handler import MongoHandler
from utils import utils
from config import TRADING_ENV


def clean_mongodb_log(db_name, backward_days=90):
    now = datetime.datetime.utcnow().date()
    client = utils.get_mgdb_client()
    db = client[db_name]
    collection_names = db.list_collection_names()
    for collection_name in collection_names:
        try:
            tem_date = datetime.datetime.strptime(collection_name, "%Y%m%d").date()
            if now - tem_date > datetime.timedelta(days=backward_days):
                db[collection_name].drop()
        except ValueError:
            pass


def get_logger(logger_name=None, fh_log_lv='debug', ch_log_lv='debug', db_name=None, use_tg=True):

    format_ = '%(levelname)s - %(asctime)s.%(msecs)03d - %(message)s - %(name)s(%(filename)s)(%(funcName)s) -line_%(lineno)d - (PID-%(process)d) (%(processName)s) (%(threadName)s)'
    time_fmt = "%H:%M:%S"
    dt = datetime.datetime.now()
    LOG_LV_DICT = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}

    if not logger_name:
        ch_log_lv = 'warning'
        fh_log_lv = 'warning'
    else:
        if TRADING_ENV == 'LIVE':
            ch_log_lv = 'info'

    ch_log_lv = LOG_LV_DICT[ch_log_lv.lower()]
    fh_log_lv = LOG_LV_DICT[fh_log_lv.lower()]

    class ColoredFormatter(logging.Formatter):
        blue = "\x1b[34;1m"
        yellow = "\x1b[33;1m"
        green = "\x1b[1;32m"
        red = "\x1b[31;1m"
        red_highlight = "\x1b[1;41m"
        reset = "\x1b[0m"

        FORMATS = {
            logging.DEBUG: blue + format_ + reset,
            logging.INFO: green + format_ + reset,
            logging.WARNING: yellow + format_ + reset,
            logging.ERROR: red + format_ + reset,
            logging.CRITICAL: red_highlight + format_ + reset
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, time_fmt)
            return formatter.format(record)

    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    logger.setLevel(fh_log_lv)

    col_name = utils.get_log_col_name(dt)

    if TRADING_ENV == 'STAGE':
        db_name = f'{TRADING_ENV}_{db_name}'

    tele_bot = None
    if use_tg:
        try:
            from utils.tg import telegram_bot
            tele_bot = use_tg if not isinstance(use_tg, bool) else telegram_bot()
        except ImportError:
            try:
                from telegram_risk_msg import telegram_bot
                tele_bot = use_tg if not isinstance(use_tg, bool) else telegram_bot()
            except ImportError:
                pass

    handler = MongoHandler(host='localhost', level=fh_log_lv, database_name=db_name, collection_name=col_name, tele_bot=tele_bot)

    clean_mongodb_log(db_name)

    logger.addHandler(handler)
    colored_formatter = ColoredFormatter()
    ch = logging.StreamHandler()
    ch.setLevel(ch_log_lv)
    ch.setFormatter(colored_formatter)
    logger.addHandler(ch)

    if logger_name:
        logger.propagate = False

    return logger
