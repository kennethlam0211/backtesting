import os
import sys
import json
import datetime
import time
import copy
import threading
import traceback
import inspect
from collections import deque
from functools import partial

import telegram
import requests
import urllib3
import zmq
import pandas as pd
from telegram import ParseMode
from telegram.error import RetryAfter
from ratelimit import limits, sleep_and_retry, RateLimitException
from rich.traceback import install
from rich.console import Console

install()
console = Console()

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils import utils
from utils.utils import check_is_running, get_col
from utils.logger import get_logger
from config import TG_CONFIG, TRADING_ENV, TG_CLIENTS_ID_DICT
import config

TG_CONFIG = TG_CONFIG[TRADING_ENV.upper()]
ONE_MINUTES = 60
MAX_TEXT_LEN = 1000

def rate_limit_factory(calls, period):
    def decorator(func):
        rate_limited_func = limits(calls=calls, period=period)(func)
        return sleep_and_retry(rate_limited_func)
    return decorator

class telegram_bot:
    # CLEAR_SET_DICT = {"client":50,"monitor":60}
    ADDR = "tcp://127.0.0.1:"
    OS_DICT = {'win32':'w','linux':'l'}
    def __init__(self,role="sender",**kwargs):
        self.log = None
        self.port = 45590
        # self.emergency_call = emergency_call()  # removed — risk_call not used
        self.os = self.OS_DICT[sys.platform]
        self.__dict__.update(kwargs)
        self.last_send_ts =  time.monotonic()
        self.items = ["client","monitor","systemctl"]
        self.bots = {}
        self.bots_in_use = {}
        self.msg_box = {}
        self.create_bots()
        self.update_id = -1
        self.create_logger()
        self.init_role(role)

    def create_logger(self):
        if not self.log:
            self.log = get_logger(logger_name='TG', fh_log_lv='error', ch_log_lv='error',db_name='LOG_DB',use_tg= False)

    def init_receiver(self):
        self.receiver = self.context.socket(zmq.SUB)
        for n in range(0,10):
            self.receiver.connect(self.ADDR+str(self.port+n))
            time.sleep(0.1)
        self.receiver.setsockopt_string(zmq.SUBSCRIBE, '')
        self.poller = zmq.Poller()
        self.poller.register(self.receiver, zmq.POLLIN)

    def init_msg_box(self):
        for item in self.items:
            self.msg_box[f'{item}_{item}'] = []
            if item != 'monitor':
                for user in config.TG_CLIENTS_ID_DICT.values():
                    self.msg_box[f'{item}_{user}'] = []
            
    def msg_sender(self,msg,bot_name,where):
        try:
            self.rotate_bots()
            self.bots_in_use[bot_name].send_message(TG_CONFIG[where],msg)
            time.sleep(2)
        except Exception as e:
            # self.log.debug(e)
            print(bot_name,where,msg,print(str(e)))
            time.sleep(10)
            self.msg_sender(msg,bot_name,where)
            # console.print_exception()
    
    def init_bot_senders(self):
        for item in self.msg_box:
            bot_name, where = item.split('_')
            obj = rate_limit_factory(20,ONE_MINUTES)(partial(self.msg_sender,bot_name=bot_name, where=where))
            setattr(self,item,obj)

    def init_raw_msg_receiver_thd(self):
        init_sleep = 10
        for item in self.msg_box:
            bot_name, where = item.split('_')
            threading.Thread(target=self.raw_msg_receiver,name=item,args=(bot_name, where, init_sleep),daemon=True).start()
            init_sleep+=10

    def init_role(self,role):
        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUB)
        for n in range(0,10):
            try:
                self.sender.bind(self.ADDR+str(self.port+n))
                time.sleep(0.1)
                break
            except zmq.error.ZMQError:
                pass
        
        if role == 'receiver':
            self.init_receiver()
            now = time.monotonic()
            self.lock = threading.Lock()
            self.init_msg_box()
            self.init_bot_senders()
            self.init_raw_msg_receiver_thd()
            # for i,item in enumerate(self.items):
            #     setattr(self,f'{item}_last_clear_ts',now)
            #     setattr(self,f'{item}_msgs',[])
            #     setattr(self,f'{item}_last_check_ts',now+i)
            #     if item not in ['systemctl']:
            #         setattr(self,f'{item}_msg_set',deque(maxlen=20))
            self.receiving_msg()

    def raw_msg_receiver(self,bot_name,where,init_sleep):

        time.sleep(init_sleep)

        while True:
            
            if self.msg_box[f'{bot_name}_{where}']:

                chatroom_msgs =  copy.deepcopy(self.msg_box[f'{bot_name}_{where}'])

                with self.lock:
                    self.msg_box[f'{bot_name}_{where}'] = self.msg_box[f'{bot_name}_{where}'][len(chatroom_msgs):]

                op_msgs = self.combine_msgs(utils.remove_duplicates(chatroom_msgs)) 

                sender_ = getattr(self,f'{bot_name}_{where}')

                for op_msg in op_msgs: 
                    if op_msg:
                        sender_(self.os+' '+op_msg.replace('"',''))

            if bot_name == 'client':
                time.sleep(50)
            else:
                time.sleep(5)
                
    # @check_is_running
    # def check_take_action(self):
    #     try:
    #         if datetime.time(8,45,0)>datetime.datetime.now().time()>datetime.time(8,45,10):
    #             time.sleep(11)
    #             exit()
    #         now = time.monotonic()
    #         for item in self.items:
    #             if now-getattr(self,f'{item}_last_check_ts') >10 or item=='systemctl':
    #                 self.send_msgs(item)
    #                 setattr(self,f'{item}_last_check_ts',now)
    #                 if item != 'systemctl':
    #                     last_clear_ts = getattr(self,f'{item}_last_clear_ts') 
    #                     if now - last_clear_ts > self.CLEAR_SET_DICT.get(item):
    #                         getattr(self,f'{item}_msg_set').clear() 
    #                         setattr(self,f'{item}_last_clear_ts',now)
    #     except Exception as e:
    #         # self.log.exception(e)
    #         console.print_exception()
            
    def receiving_msg(self):
        BOOL= True
        to_restart = False
        while BOOL:
            now = datetime.datetime.now()
            if (datetime.time(9,0,0)< now.time()<datetime.time(9,2,0)) or (datetime.time(16,45,0)< now.time()<datetime.time(16,47,0)):
                to_restart = True
            else:
                if to_restart:
                    BOOL = False
            socks = dict(self.poller.poll(3000))
            try:
                if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
                    reply = self.receiver.recv_string()
                    rev_msg = json.loads(reply)
                    rev_msg['msg'] = json.dumps(rev_msg['msg'],default=utils.dump_hook)
                    msg = rev_msg.get('msg')
                    where = rev_msg.get('where')
                    where_to = rev_msg.get('where_to') if rev_msg.get('where_to') else where
                    # mode_ = rev_msg.get('mode')
                    if not datetime.time(4,0,0)<datetime.datetime.now().time()<datetime.time(8,45,0):
                        self.msg_box[f'{where}_{where_to}'].append(msg)
                    # if mode_ or (where_to != where and  where == 'systemctl'):
                    #     try:
                    #         time.sleep(1.1)
                    #         getattr(self,f'send_msg_{where}')(self.os+' ' +msg,where,TG_CONFIG[where_to],mode = ParseMode.HTML if mode_ else None)
                    #         # if  where_to != where:
                    #         #     time.sleep(1.1)
                    #         #     getattr(self,f'send_msg_{where}')(self.os+' ' +msg,where,TG_CONFIG[where],mode =ParseMode.HTML if mode_ else None)
                    #     except Exception as e:
                    #         console.print_exception()
                    #         self.update_bot_in_use(where)
                    #         try:
                    #             getattr(self,f'send_msg_{where}')(self.os+' ' +msg,where,TG_CONFIG[where_to],mode = ParseMode.HTML if mode_ else None)
                    #         except:
                    #             console.print_exception()
                    # if not (where_to in config.TG_CLIENTS_ID_DICT.values() and where == 'systemctl') :
                    #     if msg:
                    #         if not mode_:
                    #             item_msgs = getattr(self,f'{where}_msgs')
                    #             if where in ['systemctl']:
                    #                 item_msgs.append(msg)
                    #             elif where in ["monitor","client"]:
                    #                 item_msg_set = getattr(self,f'{where}_msg_set')
                    #                 if msg not in item_msg_set:
                    #                     item_msgs.append(msg)
                    #                     item_msg_set.append(msg)
                    #         else:
                    #             time.sleep(1.1)
                    #             getattr(self,f'send_msg_{where}')(self.os+' ' +msg,where,TG_CONFIG[where_to],mode =ParseMode.HTML if mode_ else None)
                    # self.check_take_action()
                
                if self.receiver.getsockopt(zmq.EVENTS) & zmq.POLLERR:
                    self.receiver.setsockopt(zmq.LINGER, 0)
                    self.receiver.close()
                    self.poller.unregister(self.receiver)
                    time.sleep(1.1)
                    self.init_receiver()
            except Exception as e:
                console.print_exception()

    # @sleep_and_retry
    # @limits(calls=16, period=ONE_MINUTES)
    # def send_msg_systemctl(self,msg,item,where,mode=None):
    #     try:
    #         time.sleep(1.1)
    #         self.bots_in_use[item].send_message(where,msg,parse_mode=mode)
    #     except Exception as e:
    #         self.log.exception(e)
    #         console.print_exception()
    #         try:
    #             self.update_bot_in_use(item)
    #             self.bots_in_use[item].send_message(where,msg,parse_mode=mode)
    #         except:
    #             print(item,msg)
    #             console.print_exception()

    # @sleep_and_retry
    # @limits(calls=16, period=ONE_MINUTES)
    # def send_msg_client(self,msg,item,where,mode=None):
    #     try:
    #         # time.sleep(1.1)
    #         self.bots_in_use[item].send_message(where,msg,parse_mode=mode)
    #     except Exception as e:
    #         self.log.exception(e)
    #         console.print_exception()
    #         try:
    #             self.update_bot_in_use(item)
    #             self.bots_in_use[item].send_message(where,msg,parse_mode=mode)
    #         except:
    #             print(item,msg)
    #             console.print_exception()
                
    # @sleep_and_retry
    # @limits(calls=16, period=ONE_MINUTES)
    # def send_msg_monitor(self,msg,item,where,mode=None):
    #     try:
    #         time.sleep(1.1)
    #         self.bots_in_use[item].send_message(where,msg,parse_mode=mode)
    #     except Exception as e:
    #         self.log.exception(e)
    #         console.print_exception()
    #         try:
    #             self.update_bot_in_use(item)
    #             self.bots_in_use[item].send_message(where,msg,parse_mode=mode)
    #         except:
    #             print(item,msg)
    #             console.print_exception()


    # @sleep_and_retry
    # @limits(calls=16, period=ONE_MINUTES)
    # def send_msg_report(self,msg,item,where,mode=None):
    #     try:
    #         time.sleep(1.1)
    #         self.bots_in_use[item].send_message(where,msg,parse_mode=mode)
    #     except:
    #         console.print_exception()
          
    # def send_msgs(self,chatroom_name):
    #     try:
    #         chatroom_msgs = getattr(self,f'{chatroom_name}_msgs')

    #         if chatroom_msgs:

    #             tem_chatroom_msgs = chatroom_msgs

    #             msg_len = len(tem_chatroom_msgs)
                
    #             send_msgs_to_chat_room = getattr(self,f'send_msg_{chatroom_name}')
    
    #             op_msgs = self.combine_msgs(tem_chatroom_msgs) 

    #             for op_msg in op_msgs: 
    #                 if op_msg:
    #                     send_msgs_to_chat_room(self.os+' '+op_msg.replace('"',''),chatroom_name,TG_CONFIG[chatroom_name])
                
    #             [chatroom_msgs.pop(0) for i in range(msg_len)]
    
    #     except RateLimitException :
    #         self.emergency_call.make_phone_call('tg_error')
    #     except Exception as e:
    #         console.print_exception()

    def combine_msgs(self,msgs):
        try:
            op_msgs = []
            op_msg = ""
            for i,msg in enumerate(msgs):
                if len(msg)>MAX_TEXT_LEN:
                    if op_msg:
                        op_msgs.append(op_msg)
                        op_msg = ""
                    iterations =  (len(msg) //MAX_TEXT_LEN )+1
                    for index,iteration in enumerate(range(iterations)):
                        if index ==0:
                            new_msg = f'{i}. '
                        else:
                            new_msg = ''
                        new_msg += msg[MAX_TEXT_LEN*index:MAX_TEXT_LEN*(index+1)]

                        op_msgs.append(new_msg)
                else:
                    if len(op_msg+msg) < MAX_TEXT_LEN:
                        op_msg += f"{i}. "+msg +"\n" +"\n"
                    else:
                        op_msgs.append(op_msg)
                        op_msg = f"{i}. "+msg +"\n" +"\n"
            if op_msg:
                op_msgs.append(op_msg)
            return op_msgs
        except Exception as e:
            console.print_exception()

    def create_bots(self):
        try:
            bot_names=['systemctl','monitor','spare_systemctl','spare_monitor','report','client','spare_client']
            for bot_name in bot_names:
                self.bots[bot_name] = telegram.Bot(token=TG_CONFIG['bots'][bot_name])
            self._current_shift = None
            self.rotate_bots()
        except Exception as e:
            console.print_exception()

    def rotate_bots(self):
        """Swap between main and spare bots every 12 hours based on current time."""
        shift = '' if datetime.time(9,0,0) < datetime.datetime.now().time() < datetime.time(21,0,0) else 'spare_'
        if shift != self._current_shift:
            self._current_shift = shift
            for item in ["client", "monitor", "systemctl"]:
                self.bots_in_use[item] = self.bots[f'{shift}{item}']



        #     print(self.bots[item]['first_name'])
        # except Exception as e:
        #     self.log.exception(e)
        #     self.bots_in_use[item] = self.bots[f'spare_{item}']
        # try:
        #     self.bots[f'spare_{item}']['first_name']
        # except Exception as e:
        #     self.bots_in_use[item] = self.bots[item]
        #     self.log.exception(e)
        # time.sleep(5)

    def get_bot(self,token):
        bot = telegram.Bot(token=token)
        # print(f'result of get_bot(): {bot.get_me()}')

    def get_bot_info(self,token): #offset = update_id of last processed update + 1 # -1 == reset
        try:
            url = f'https://api.telegram.org/bot{token}/getUpdates?offset={self.update_id}'
            res = requests.get(url,timeout=15)
            return res.json()
        except:
            pass

    def receive_tg_msg(self):
        res = None
        bot_id_mapping = {'5934662397':'systemctl','5485456962':'systemctl','6249441086':'spare_systemctl','5608781180':'spare_systemctl'}
        try:
            token_in_use = TG_CONFIG['bots'].get(bot_id_mapping.get(str(self.bots_in_use['systemctl']['id'])))
            res = self.get_bot_info(token_in_use)
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout, requests.exceptions.SSLError,
                urllib3.exceptions.MaxRetryError, urllib3.exceptions.ReadTimeoutError,telegram.error.NetworkError) as e:
            # self.update_bot_in_use('systemctl')
            # for cli in config.TG_CLIENTS_ID_DICT.values():
            #     self.send_tg_msg('Hi, Im on rata',where='systemctl',cli=cli)
            self.log.critical(str(e)) 
            time.sleep(5)
        except Exception as e :
            self.log.critical(str(e))
            time.sleep(5)
            # self.update_bot_in_use('systemctl')
            # res = self.get_bot_info(token_in_use)
            # try:
            #     token_in_use = TG_CONFIG['bots'].get(bot_id_mapping.get(str(self.bots_in_use['systemctl']['id'])))
            # except Exception as e :
            #     self.log.critical(e)
        update_id = None
        if res and res.get('result'):
            now = datetime.datetime.now()
            texts = {}
            for msg_ in res['result']:
                msg = msg_.get('message')
                if not update_id:                    
                    self.update_id = msg_.get('update_id')+1
                if msg:
                    where = msg.get('from').get('id')
                    if where in config.TG_CLIENTS_ID_DICT:
                        ts = utils.unix_to_datetime(msg.get('date'))
                        if now-datetime.timedelta(seconds=self.check_freq) >ts or list(texts) == list(config.TG_CLIENTS_ID_DICT):
                            break
                        elif msg.get('text') and where not in texts:
                            texts.update({config.TG_CLIENTS_ID_DICT[where]:msg.get('text')})
            return texts

    def msg_to_str(self,msg):
        try:
            if msg is None:
                return ''
            if isinstance(msg,dict):
                msg.pop('_id',None)
            return json.dumps(msg,default=utils.dump_hook)
        except Exception as e:
            console.print_exception()
    
    def send_tg_msg(self,msg,where="monitor",mode='',cli=None,cli_only=False):
        try:

            if not msg:
                msg = 'empty_msg'
                self.log.critical(msg)

            if where == "client" and isinstance(msg,dict):  
                msg = msg.pop("CRITICAL",msg)
                msg.pop('tag',None)
                
            where_to = cli if cli else where

            if where == "report":
                msg = msg.replace('"','')
                mode = mode if mode == '' else ParseMode.HTML
                self.bots[where].send_message(TG_CONFIG[where],self.os+' '+msg,parse_mode=mode)
                if cli:
                    self.bots[where].send_message(TG_CONFIG[where_to],self.os+' '+msg,parse_mode=mode)
                    time.sleep(1.5)
            else:
                if where != where_to and where!='systemctl':
                    msg_ =self.msg_to_str({"msg":msg,"where":where,'where_to':where})
                    self.sender.send(msg_.encode()) 
                if not cli_only:
                    msg_ =self.msg_to_str({"msg":msg,"where":where,'where_to':where_to})
                    self.sender.send(msg_.encode()) 

            # if cli in config.TG_CLIENTS_ID_DICT.values() and where != 'systemctl':
            #     msg_ =self.msg_to_str({"msg":msg,"where":where,"mode":mode,'where_to':where})
            #     self.sender.send(msg_.encode()) 

        except Exception as e:        
            # self.log.exception(e)
            console.print_exception()



if __name__ == '__main__':
    pass
    # token = "2080447270:AAGBFnSrY2cS2kpcxqQqvjhygeIKuTTB9n4"
    # telegram_bot().receive_tg_msg()
    # df = pd.DataFrame({'a':[1,2,3],'b':[23,45,4]})
    # message = df.to_string()
    # # message = 'This is an underlined message:' +'\n<b><u>Hello</u></b>'
    # # message = 'This is an underlined message: \n' + "\u0332" + "H" + "\u0332" + ' '+'<b><u>' + 'c' + '</u></b>'
    # # res = telegram_bot().receive_tg_msg()
    # # print(res)
    # # telegram_bot().get_bot_info(token='5934662397:AAF8WJDeBJ2b5FQgXZo2bxmZxGEtpBZV_UI')
    # telegram_bot().bots['spare_monitor'].send_message('-1001524762202','33333333333333333',parse_mode=ParseMode.HTML)
   

    # import requests
    # TOKEN = "YOUR TELEGRAM BOT TOKEN"
    # chat_id = "YOUR CHAT ID"
    # message = "hello from your telegram bot"
    # url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    # print(requests.get(url).json()) # this sends the message