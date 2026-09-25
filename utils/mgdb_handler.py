import datetime as dt
import logging
import inspect
import traceback
import time

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure, PyMongoError, ServerSelectionTimeoutError

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils import utils

write_method = 'insert_one'
write_many_method = 'insert_many'

_connection = None


class MongoFormatter(logging.Formatter):

    DEFAULT_PROPERTIES = logging.LogRecord('', '', '', '', '', '', '', '').__dict__.keys()

    def format(self, record):
        try:
            tem = dt.datetime.now()
            inspect_ = inspect.stack(context=1)
            caller = []
            try:
                for i in reversed(range(len(inspect_))):
                    if record.funcName == inspect_[i].function:
                        break
                    caller.extend(inspect_[i].code_context)
            except Exception:
                pass

            document = {
                'cts': tem,
                'level': record.levelname,
                'threadId': record.thread,
                'threadName': record.threadName,
                'levelno': record.levelno,
                'loggerName': record.name,
                'path': record.pathname,
                'script': record.module,
                'func': record.funcName,
                'lineno': record.lineno,
                'processName': record.processName,
                'pid': record.process,
                'caller_func': caller,
            }

            if record.exc_info is not None:
                exception_dict = {
                    'error_message': str(record.exc_info[1]),
                    'more': str(record.exc_info[0]),
                    'stackTrace': self.formatException(record.exc_info)
                }
                if hasattr(record, 'extra_msg'):
                    exception_dict.update(record.extra_msg)
                document.update({'tag': 'exception', **exception_dict})
            else:
                exception_dict = None
                if isinstance(record.msg, dict):
                    document.update(record.msg)
                else:
                    document.update({'msg': record.msg})

            return document, exception_dict

        except Exception:
            utils.get_exception_msg()


class MongoHandler(logging.Handler):

    def __init__(self, level=logging.NOTSET, host='localhost', port=27017,
                 database_name='logs', collection_name='logs',
                 username=None, password=None, authentication_db='admin',
                 fail_silently=False, formatter=None, capped=False,
                 capped_max=1000, capped_size=1000000, reuse=True, tele_bot=None, **kwargs):
        logging.Handler.__init__(self, level)
        self.host = host
        self.port = port
        self.database_name = database_name
        self.collection_name = collection_name
        self.username = username
        self.password = password
        self.authentication_database_name = authentication_db
        self.fail_silently = fail_silently
        self.connection = None
        self.db = None
        self.collection = None
        self.authenticated = False
        self.tele_bot = tele_bot
        self.formatter = formatter or MongoFormatter()
        self.capped = capped
        self.capped_max = capped_max
        self.capped_size = capped_size
        self.reuse = reuse
        self._connect(**kwargs)

    def _connect(self, **kwargs):
        global _connection
        if self.reuse and _connection:
            self.connection = _connection
        else:
            self.connection = utils.get_mgdb_client()
            try:
                self.connection.is_primary
            except ServerSelectionTimeoutError:
                if self.fail_silently:
                    return
                else:
                    raise
            _connection = self.connection

        self.db = self.connection[self.database_name]

        if self.capped:
            try:
                self.collection = Collection(self.db, self.collection_name,
                                             capped=True, max=self.capped_max,
                                             size=self.capped_size)
            except OperationFailure:
                self.collection = self.db[self.collection_name]
        else:
            self.collection = self.db[self.collection_name]

    def close(self):
        if self.authenticated:
            self.db.logout()
        if self.connection is not None:
            self.connection.close()

    def emit(self, record):
        if self.collection is not None:
            try:
                col_name = utils.get_log_col_name()
                if col_name != self.collection_name:
                    self.collection_name = col_name
                    self.collection = utils.get_col(db_name=self.database_name, col_name=self.collection_name)

                final_msg, exception_dict = self.format(record)
                getattr(self.collection, write_method)(final_msg)

                if final_msg["level"] in ["ERROR", "CRITICAL"] and self.tele_bot:
                    msg_destination = final_msg.get("where", "monitor")
                    if exception_dict is not None:
                        self.tele_bot.send_tg_msg({'exception': exception_dict})
                    else:
                        if isinstance(record.msg, dict):
                            record.msg.pop('where', None)
                        self.tele_bot.send_tg_msg({final_msg["level"]: record.msg}, where=msg_destination)

            except Exception:
                if not self.fail_silently:
                    self.handleError(record)
                    utils.get_exception_msg()

    def __exit__(self, type, value, traceback):
        self.close()
