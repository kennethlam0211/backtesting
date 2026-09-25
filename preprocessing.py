import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import List, Optional
import numpy as np
import datetime
import time
from params import *
from collections import deque


# def session_preprocessing(df,first_open):
#     df = to_normailse(df,first_open) 
#     df = arrange_ohlc(df) 
#     return df

#################################################################################


# def arrange_ohlc(df):
#     hl = df["hl"].values  
#     open_px = df["open"].values
#     high_px = df["high"].values
#     low_px = df["low"].values
#     close_px = df["close"].values
#     avg_px = df["avg_px"].values
    
#     # Create output array efficiently
#     out = np.column_stack([
#                             open_px,
#                             np.where(hl==1, high_px, low_px),
#                             np.where(hl==1, low_px, high_px),
#                             close_px
                            
#                           ])
    
#     df["ohlc_vec"] = out.tolist()
#     return df




# def to_normailse_gp(gp,ohlc_norm_factor,first_open=None,norm_cols=None):
#     if not gp.empty:
#         if not first_open:
#             use_open = 'open' if 'open' in gp.columns else 'open_1'
#             first_open = gp[use_open].iloc[0]
#         if norm_cols is None:
#             price_cols = ['open', 'high', 'low', 'close','avg_px']
#             gp[[f"n_{c}" for c in price_cols]] = gp[price_cols].sub(first_open).div(ohlc_norm_factor)
#         else:
#             norm_cols_ = [f'{col}_' for col in norm_cols]
#             gp[norm_cols] = gp[norm_cols].sub(first_open).div(ohlc_norm_factor)
#         # gp['long_reward'] = -gp['n_open'] + gp['n_close']
#         # gp['short_reward'] = -gp['long_reward']
#         # B = df['n_open']
#         # C = df['n_high']
#         # D = df['n_low']
#         # gp['bar_score'] = np.maximum(C - B, B - D)- np.minimum(C - B, (B - D) * 2) - np.abs(gp['short_reward'])
#         # gp['acc_long_reward'] = gp['long_reward'].cumsum()
#         # gp['acc_short_reward'] = gp['short_reward'].cumsum()
#         # gp['acc_bar_score'] = gp['bar_score'].cumsum()
#         return gp

# def to_normailse(df,freq,norm_cols=None):
#     ohlc_norm_factor = OHLC_NORM_FACTORS[freq]
#     df = df.groupby(pd.Grouper(freq='720min',key='ts'),group_keys=False).apply(lambda x: to_normailse_gp(x, ohlc_norm_factor, norm_cols=norm_cols))
#     df = df[df['ts'].isnull()==False]
#     df.reset_index(drop=True, inplace=True)
#     return df

# def get_session_open(df):
#     df['session_open'] = df['open'].groupby(pd.Grouper(freq='720min',key='ts'),group_keys=False)['open'].transform('first')
#     return df


def normalize_vix_ewma(series, span=34000):
    log_vix = np.log(series)
    ewma_mean = log_vix.ewm(span=span).mean()
    ewma_std = log_vix.ewm(span=span).std()
    return (log_vix - ewma_mean) / ewma_std




def get_candle(df,threshold=0.9):
    df['large_candle'] = 0.0
    bar_range = df['bar_range'].replace(0,pd.NA)
    df.loc[(df['large_candle'].abs()>threshold) & ( df['bar_diff'].abs()>df['20_std']*2) & (df['bar_range']>0),'large_candle'] = df['bar_diff']/ bar_range
    return df

def get_needle(df,threshold = 0.25):
    df['needle'] = 0.0
    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']
    neelde_range = np.maximum(np.minimum(O-L,C-L),0.01)
    needle_body = np.maximum(H-O,H-C)
    df.loc[((needle_body/neelde_range)<threshold)&(neelde_range>df['20_std']),'needle']= 1-(needle_body/neelde_range)
    neelde_range = np.maximum(np.minimum(H-O,H-C),0.01)
    needle_body = np.maximum(O-L,C-L)
    df.loc[((needle_body/neelde_range)<threshold)&(neelde_range>df['20_std']),'needle']= (1-(needle_body/neelde_range))*-1
    return df

def get_bar_score(df):
    df['long_reward'] = -df['open'] + df['close']
    #df['short_reward'] = -df['long_reward']
    B = df['open']
    C = df['high']
    D = df['low']
    df['bar_score'] = (np.maximum(C - B, B - D)- np.minimum(C - B, B - D) *2 + np.abs(df['long_reward']))/ NORM_FACTOR
    df.drop(columns=['long_reward'], inplace=True)
    return df

def get_high_low_targets(df):
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['cum_high'] = df.groupby(pd.Grouper(freq='720min', key='ts'))['high'].transform('cummax')
    df['cum_low'] = df.groupby(pd.Grouper(freq='720min', key='ts'))['low'].transform('cummin')
    #df['intrady_day_hl_targets'] = list(zip((df['cum_high']-df['close'])/NORM_FACTOR,(df['cum_low']-df['close'])/NORM_FACTOR)) 
    df['intrady_day_hl_targets'] = list(np.column_stack([df['cum_high'].to_numpy(),df['cum_low'].to_numpy()]))
    df.drop(columns=['cum_high','cum_low'], inplace=True)
    return df


def get_ref_px(df):
    df['ref_px'] = df.groupby(pd.Grouper(freq='720min', key='ts'))['open'].transform('first').shift(-1).ffill()
    return df

def get_pre_ohlc(df):
    df['pre_ohlc_vec'] = df['ohlc_vec'].shift(1)
    df['pre_ohlc'] = df.apply(lambda row: np.array(row['pre_ohlc_vec'] + row['ohlc_vec']) if row['pre_ohlc_vec'] else pd.NA, axis=1)
    df['pre_ohlc_targets'] = list(df['pre_ohlc'].to_numpy())
    #df['pre_ohlc'] = [[(x - close)/norm_factor for x in arr] for arr, close in zip(df['pre_ohlc'], df['close'])]
    df.drop(columns=['pre_ohlc_vec','pre_ohlc'], inplace=True)
    return df


def get_sma_targets(df):
    sma_targets = []
    for window in [20,50,100]:
        df[f'sma_{window}_day']= df.loc[df['rth']!=1,'close'].rolling(window=window).mean()
        df[f'sma_{window}_day'].ffill(inplace=True)
        df[f'sma_{window}_all']= df['close'].rolling(window=window).mean()
        sma_targets.append(f'sma_{window}_day')
        sma_targets.append(f'sma_{window}_all')
    df['sma_targets'] = df.apply(lambda row: np.array([row[item] for item in sma_targets]) if not row[sma_targets].isna().any() else pd.NA, axis=1)    
    df['sma_targets'] = list(df['sma_targets'].to_numpy())
    df.drop(columns=sma_targets, inplace=True)
    return df

def preprocessing(df,freq):
    df = get_bar_score(df)
    if freq=='720':
        df = get_pre_ohlc(df)
        df = get_sma_targets(df)
    return df

#############################################TAS###########################################
def unit_bbands(px_list,window):
    px_list = px_list[-window:]
    std = np.std(px_list)
    mean = np.mean(px_list)
    px = px_list[-1]
    return (px-(mean-2*std))/(4*std)

def unit_bband_targets(px_list,window):
    px_list = px_list[-window:]
    std = np.std(px_list)
    mean = np.mean(px_list)
    return np.array([mean,mean+std*2,mean-std*2],dtype=np.float32)


def bbands(df: pd.DataFrame, window: int = 20, dev_factor: float = 2.0) -> pd.DataFrame:
    df[f'{window}_Upper_band'] = df[f'{window}_sma']+df[f'{window}_std']*dev_factor*2
    df[f'{window}_Lower_band'] = df[f'{window}_sma']-df[f'{window}_std']*dev_factor*2
    df[f'{window}_bbands'] = (df['close']- df[f'{window}_sma'])/(dev_factor*2*df[f'{window}_std'])

    cols = [f'{window}_sma',f'{window}_Upper_band',f'{window}_Lower_band']
    df[f'{window}_bband_targets'] = list(df[cols].to_numpy())
#     df[f'{window}_bband_targets_n'] = df[f'{window}_bband_targets'].apply(
#     lambda t: tuple(x / 5 for x in t) if not isna(t) else pd.NA
# )
    df.drop(columns=[f'{window}_sma',f'{window}_Upper_band',f'{window}_Lower_band'], inplace=True)
    return df

def unit_std(px_list,window):
    px_list = px_list[-window:]
    return np.std(px_list)

# n_close, close
def std(df,window,px_col='close'):
    df[f'{window}_std'] = df[px_col].rolling(window=window).std(ddof=0) 
    return df

def sma(df,window):
    df[f'{window}_sma'] = df['close'].rolling(window=window).mean()
    return df


def atr(df: pd.DataFrame, window: int = 14, 
        high_col: str = 'high', low_col: str = 'low', close_col: str = 'close',
        ) -> pd.DataFrame:
    """
    Vectorized Wilder's Average True Range (ATR)
    Adds ATR column to the DataFrame (or returns Series if output_col=None)
    
    Similar style to your sma_rsi function.
    """
    period = 14 
    if not all(col in df.columns for col in [high_col, low_col, close_col]):
        raise ValueError(f"DataFrame must contain columns: {high_col}, {low_col}, {close_col}")

    high  = df[high_col]
    low   = df[low_col]
    close = df[close_col]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (EMA with com = period-1, adjust=False)
    atr_series = tr.ewm(
        com=period - 1,
        min_periods=period,
        adjust=False
    ).mean()

    # Decide output
    
    df[f'{window}_atr'] = atr_series
    return df

def unit_atr(prices_high: List[float], 
             prices_low: List[float], 
             prices_close: List[float], 
             window: int = 14) -> Optional[float]:
    """
    Calculate the most recent ATR value given lists of high, low, close prices.
    Uses Wilder's smoothing logic.
    
    Expects the last (period + 1) values at minimum.
    Returns the current (latest) ATR value or None if not enough data.
    """
    hard_window=  14
    n = hard_window
    if (len(prices_high) < n + 1 or 
        len(prices_low)  < n + 1 or 
        len(prices_close)< n + 1):
        return None

    # Take last n+1 elements
    highs  = prices_high[-(n+1):]
    lows   = prices_low[-(n+1):]
    closes = prices_close[-(n+1):]

    # Calculate True Ranges for the period
    trs = []
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i]  - closes[i-1])
        trs.append(max(tr1, tr2, tr3))

    if len(trs) < n:
        return None

    # First ATR = simple average of first n true ranges
    atr = sum(trs[:n]) / n

    # Then update with the last (most recent) TR using Wilder's formula
    last_tr = trs[-1]
    atr = (atr * (n - 1) + last_tr) / n

    return atr

# def unit_sma(px_list,window):
#     px_list = px_list[-window:]
#     return np.mean(px_list)-px_list[-1]

def transform_rsi_data(raw_data, clip=0.6128, k=1.6128, smooth_threshold=0.01, 
                                     soft_clip=True, eps=1e-8):
    """
    Hybrid version that can match original transform_data or use soft clipping
    
    Parameters:
    raw_data: Input array from -1 to 1
    clip: If 0.5, clip at 1.5 and -1.5
    k: Controls how fast it grows away from zero (0.1=slow, 2=fast)
    smooth_threshold: Below this value, use smooth tanh approximation for sign
    soft_clip: True for tanh soft clipping (gradient-friendly), False for hard clipping
    eps: Small value to avoid division by zero
    """
    # Calculate actual clip value
    clip_value = 1 + clip
    
    # Ensure raw_data is numpy array
    raw_data = np.asarray(raw_data)
    
    # Compute absolute value (with epsilon for stability)
    abs_x = np.abs(raw_data) + eps
    
    # Compute |x|^k
    if k == 1:
        power = abs_x
    else:
        power = np.power(abs_x, k)
    
    # Create smooth sign approximation
    sign_approx = np.zeros_like(raw_data)
    
    # Mask for small values
    mask_small = np.abs(raw_data) < smooth_threshold
    
    # Small x: use smooth approximation (tanh)
    if np.any(mask_small):
        sign_approx[mask_small] = np.tanh(raw_data[mask_small] / smooth_threshold)
    
    # Large x: use exact sign
    mask_large = ~mask_small
    if np.any(mask_large):
        sign_approx[mask_large] = np.sign(raw_data[mask_large])
    
    # Compute transformation: clip_value * sign_approx * |x|^k
    transformed = clip_value * sign_approx * power
    
    # Apply clipping
    if soft_clip:
        # Soft clipping with tanh (gradient-friendly)
        result = clip_value * np.tanh(transformed / clip_value)
    else:
        # Hard clipping (original behavior)
        result = np.clip(transformed, -clip_value, clip_value)
    
    return result

def sma_rsi(df, window):
    hard_window = 14
    df['change'] = df['close'].diff()

    df['gain'] = df['change'].clip(lower=0)
    df['loss'] = (-df['change']).clip(lower=0)

    # 3) Rolling (window) SMA of gains and losses
    df['avg_gain'] = df['gain'].rolling(hard_window).mean()
    df['avg_loss'] = df['loss'].rolling(hard_window).mean()

    # 4) Relative Strength (SMA version). If avg_loss = 0 → RS = inf → RSI = 100
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df.loc[df['avg_loss'] == 0, 'rs'] = np.inf

    # 5) RSI on 0–100 scale, then divide by 100 for 0–1
    df[f'{window}_sma_rsi'] = 100 - (100 / (1 + df['rs']))
    df[f'{window}_sma_rsi'] = ((df[f'{window}_sma_rsi'] / 100 )*2 -1)*-1
    df.drop(columns=['change','gain','loss','avg_gain','avg_loss','rs'], inplace=True)
    return df

def unit_sma_rsi(px_list: List[float], window: int) -> Optional[float]:
    hard_window = 14
    n = hard_window
    if len(px_list) < n + 1:
        return None

    recent = px_list[-(n + 1) :]                # last window+1 closes
    deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [(-d) if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n

    if avg_loss == 0:
        return 1.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = ((rsi/100)*2 -1)*-1
    return rsi

def price_session(df,window):
    df['pre_open'] = df['open'].shift(1)
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"]  = df["low"].shift(1)
    df["pre_close"] = df["close"].shift(1)
    df['HO'] = df['open'] - df['pre_open']
    df["HH"] = df["high"] - df["prev_high"]          # Higher High
    df["HL"] = df["low"]  - df["prev_low"]           # Lower  Low
    df['HC'] = df['close'] - df['pre_close']
    df['px_session'] = list(zip(df["HO"],df["HH"], df["HL"],df["HC"]))

    mask = df['hl'] == 0
    cols = ["HO", "HL", "HH", "HC"]
    df.loc[mask, 'px_session'] = df.loc[mask, cols].apply(tuple, axis=1)
    
    # df.loc[mask, 'px_session'] = list(zip(
    #                                             df.loc[mask, "HO"], 
    #                                             df.loc[mask, "HL"], 
    #                                             df.loc[mask, "HH"], 
    #                                             df.loc[mask, "HC"]
    #                                             ))
    
    df.drop(columns=['pre_open','prev_high','prev_low','pre_close','HO','HH','HL','HC'], inplace=True)
    return df

# def uni_price_session(df,freq):
#     df[[f'open_{freq}',f'high_{freq}',f'low_{freq}',f'close_{freq}']] = df[[f'open_{freq}',f'high_{freq}',f'low_{freq}',f'close_{freq}']].ffill()
#     df[f'HO_{freq}'] = df["open_1"] >= df[f"open_{freq}"]
#     df[f'HH_{freq}'] = df["high_1"] > df[f"high_{freq}"]
#     df[f'HL_{freq}'] = df["low_1"] < df[f"low_{freq}"]
#     df[f'HC_{freq}'] = df["close_1"] >= df[f"close_{freq}"]
#     df['temp'] =  list(zip(HO,HH,HL,HC))
#     df.loc[df[f"price_session_{freq}"].isna(),f"price_session_{freq}"] = df.loc[df[f"price_session_{freq}"].isna(),'temp'] 
#     df.loc[df[f'avg_px_{freq}'].isna(),[f'open_{freq}',f'high_{freq}',f'low_{freq}',f'close_{freq}']] = pd.NA
#     df.drop(columns=['temp'], inplace=True)
#     return df



# #cant use unit, otherwise keep the ema
# def get_unit_wilder_rsi(px_list: List[float], window: int) -> Optional[float]:

#     n = window
#     if len(px_list) < n + 1:
#         return None

#     recent = px_list[-(n + 1) :]
#     deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

#     gains = [d if d > 0 else 0.0 for d in deltas]
#     losses = [(-d) if d < 0 else 0.0 for d in deltas]

#     # Build a tiny DataFrame so we can use pandas' EWM with min_periods=window
#     temp = pd.DataFrame({
#                         'gain': gains,
#                         'loss': losses
#                         })

#     avg_gain_ewm = temp['gain'].ewm(alpha=1/n, min_periods=n, adjust=False).mean().iloc[-1]
#     avg_loss_ewm = temp['loss'].ewm(alpha=1/n, min_periods=n, adjust=False).mean().iloc[-1]

#     if avg_loss_ewm == 0:
#         return 1.0

#     rs = avg_gain_ewm / avg_loss_ewm
#     rsi = 100 - (100 / (1 + rs))
#     return rsi/100

# def wilder_rsi(df,window):
#     df['change'] = df['close'].diff()

#     # 2) Split into gain / loss
#     df['gain'] = df['change'].clip(lower=0)
#     df['loss'] = (-df['change']).clip(lower=0)

#     # 3) Wilder’s smoothing via EWM (α = 1/window)
#     df['avg_gain_ewm'] = df['gain'].ewm(alpha=1/window, min_periods=window, adjust=False).mean()
#     df['avg_loss_ewm'] = df['loss'].ewm(alpha=1/window, min_periods=window, adjust=False).mean()

#     # 4) Relative Strength (Wilder version)
#     df['rs_wilder'] = df['avg_gain_ewm'] / df['avg_loss_ewm']
#     df.loc[df['avg_loss_ewm'] == 0, 'rs_wilder'] = np.inf  # handle zero loss → RSI = 100

#     # 5) Standard RSI (0–100)
#     df[f'{window}_rsi'] = 100 - (100 / (1 + df['rs_wilder']))
#     df.drop(columns=['change','gain','loss','avg_gain','avg_loss','rs','rs_wilder'], inplace=True)
#     return df

def std(df,window,px_col='close'):
    df[f'{window}_std'] = df[px_col].rolling(window=window).std(ddof=0) 
    return df

def ema_std(df,window,px_col='close'):
    df[f'{window}_ema_std'] = df[px_col].ewm(span=28).std()
    return df

def UD_cal(df,px_col,window,freq,U_last=None,D_last=None):
    k = f'{window}_std_{freq}'
    U_col = f'{window}_U_{freq}'
    D_col = f'{window}_D_{freq}'
    UD_flag_col = f'{window}_UD_flag_{freq}'

    n = len(df)
    px = df[px_col].values.astype(np.float64)
    kv = df[k].values.astype(np.float64)

    U_arr = np.full(n, np.nan)
    D_arr = np.full(n, np.nan)
    flag_arr = np.full(n, -1, dtype=np.int8)

    for i in range(n):
        if i == 0:
            if px[i+10] - px[i] > 0:
                U_last = px[i]
                D_last = 0.0
            else:
                U_last = 0.0
                D_last = px[i]

        if U_last == 0 and px[i] - D_last > kv[i]:#leave_d
            U_update = px[i]
            flag_arr[i] = 1

        elif U_last == 0 or U_last - px[i] > kv[i]:#leave_u
            if i > 0:
                U_arr[i-1] = U_last
            U_update = 0.0

        elif px[i] > U_last:
            U_update = px[i]
            flag_arr[i] = 1
        else:
            U_update = U_last
            flag_arr[i] = 1

        if D_last == 0 and U_last - px[i] > kv[i]:#leave_u
            D_update = px[i]

        elif D_last == 0 or px[i] - D_last > kv[i]:#leave_d
            if i > 0:
                D_arr[i-1] = D_last
            D_update = 0.0

        elif px[i] < D_last:
            D_update = px[i]

        else:
            D_update = D_last

        U_last = U_update
        D_last = D_update

    U_arr[U_arr == 0] = np.nan
    D_arr[D_arr == 0] = np.nan

    # Shift forward by 1 to prevent look-ahead (U/D at i-1 depends on bar i's price)
    df.loc[:, U_col] = np.concatenate([[np.nan], U_arr[:-1]])
    df.loc[:, D_col] = np.concatenate([[np.nan], D_arr[:-1]])
    df.loc[:, UD_flag_col] = np.concatenate([[-1], flag_arr[:-1]])
    df.loc[:, f'{window}_UD_{freq}'] = np.nan
    return df

# def UD_cal(df,px_col,window,freq,U_last=None,D_last=None):
#     k = f'{window}_std_{freq}'
#     U_col = f'{window}_U_{freq}'
#     D_col = f'{window}_D_{freq}'
#     df[U_col] = pd.NA
#     df[D_col] = pd.NA
#     UD_flag_col = f'{window}_UD_flag_{freq}'
#     df[UD_flag_col] = -1
#     df[f'{window}_UD_{freq}'] = pd.NA
#
#     for i in range(len(df)):
#
#         if i==0:
#             if df.loc[i+10,px_col]- df.loc[i,px_col]>0:
#                 U_last = df.at[i,px_col]
#                 D_last = 0
#             else:
#                 U_last = 0
#                 D_last = df.at[i,px_col]
#
#         if U_last == 0 and df.at[i,px_col] - D_last > df.at[i,k]:#leave_d
#             U_update = df.at[i,px_col]
#             df.at[i,UD_flag_col] = 1
#
#         elif U_last == 0 or U_last - df.at[i,px_col] >  df.at[i,k]:#leave_u
#             df.at[i-1,U_col] = U_last
#             U_update = 0
#
#         elif df.at[i,px_col] > U_last:
#             U_update = df.at[i,px_col]
#             df.at[i,UD_flag_col] = 1
#         else:
#             U_update = U_last
#             df.at[i,UD_flag_col] = 1
#
#         if D_last == 0 and U_last - df.at[i,px_col] >  df.at[i,k]:#leave_u
#             D_update = df.at[i,px_col]
#
#         elif D_last == 0 or df.at[i,px_col] - D_last >  df.at[i,k]:#leave_d
#             df.at[i-1,D_col] = D_last
#             D_update = 0
#
#         elif df.at[i,px_col] < D_last:
#             D_update = df.at[i,px_col]
#
#         else:
#             D_update = D_last
#
#         U_last = U_update
#         D_last = D_update
#
#     df.loc[df[U_col]==0,U_col] = pd.NA
#     df.loc[df[D_col]==0,D_col] = pd.NA
#     return df

# def flag_period_extreme(
#         df: pd.DataFrame,
#         window: int,
#         freq: str,
#         source_col: str,
#         new_col: str,
#         mode: str = 'max'
#     ) -> pd.DataFrame:
#     source_col = f'{window}_{source_col}_{freq}'
#     new_col = f'{window}_{new_col}_{freq}'

#     # 1) Identify the blocks by cumulatively summing where source_col == 0
#     #    Each time source_col == 0, the group counter increments.
#     df['_group'] = (df[source_col] == 0).cumsum()

#     # 2) Keep only those rows where source_col != 0 (i.e. inside a period)
#     mask_nonzero = df[source_col] != 0
#     only_periods = df[mask_nonzero].copy()

#     if mode == 'max':
#         # 3a) For each group, find the index of the first maximum source_col
#         extreme_idx = only_periods.groupby('_group')[source_col].idxmax().values
#     elif mode == 'min':
#         # 3b) For each group, find the index of the first minimum source_col
#         extreme_idx = only_periods.groupby('_group')[source_col].idxmin().values
#     else:
#         raise ValueError("mode must be either 'max' or 'min'")

#     # 4) Create the new column, defaulting to 0
#     df[new_col] = 0

#     # 5) At the extreme indices, copy the original source_col value (or you could set 1 if you prefer a flag)
#     df.loc[extreme_idx, new_col] = df.loc[extreme_idx, source_col]
#     df.loc[df[new_col]!=0,new_col] = 1

#     if 'U' in source_col:
#         df[f'{window}_UD_{freq}_'] = 1
#     else:
#         df.loc[df[source_col]!=0, f'{window}_UD_{freq}_'] = 0
        
#     df.loc[df[new_col]!=1,source_col] = pd.NA
#     df.loc[df[new_col]==1,source_col] = df[source_col]
#     #6) Clean up the helper column
#     df.drop(columns=['_group',new_col], inplace=True)
#     return df

def fvg(df,window, high_col='high', low_col='low', close_col='close',
               min_candle_multiplier=1.5, require_momentum=True):
    df = df.copy()
    high = df[high_col].values
    low = df[low_col].values

    c1_high = high[:-2]
    c1_low = low[:-2]
    c3_high = high[2:]
    c3_low = low[2:]

    bullish = c3_low > c1_high
    bearish = c3_high < c1_low

    if require_momentum:
        c2_close = df[close_col].values[1:-1]
        c2_open = df['open'].values[1:-1]
        bullish &= c2_close > c2_open
        bearish &= c2_close < c2_open

    if min_candle_multiplier > 0:
        c1_size = c1_high - c1_low
        c2_size = high[1:-1] - low[1:-1]
        c3_size = c3_high - c3_low
        avg_size = (c1_size + c2_size) / 2
        size_ok = c3_size >= min_candle_multiplier * avg_size
        bullish &= size_ok
        bearish &= size_ok

    fvg = np.zeros(len(df), dtype=np.int8)
    fvg[2:][bullish] = 1
    fvg[2:][bearish] = -1
    df[f'{window}_fvg'] = fvg
    return df

def get_UD_targets(df):
    windows_mapping ={
                        '1': 500,
                        '15': 1080,
                        '30': 2160,
                        '60': 5400,
                        '720': 30800,
                        'W': 152400
                    }
    
    def padding_zero(arr,length=20):
        if arr.shape[0] < length:
            return np.pad(arr, (length - arr.shape[0], 0), 'constant')
        return arr

    for freq in FREQS: 
        
        target_window = windows_mapping[freq] *2
        #norm_factor = NORM_FACTOR #NORM_FACTOR[freq] 
        df[f'{max(WINDOWS)}_UD_{freq}_'] = df[f'{max(WINDOWS)}_U_{freq}'].fillna(0) + df[f'{max(WINDOWS)}_D_{freq}'].fillna(0)
        df.loc[df[f'{max(WINDOWS)}_UD_{freq}_'] == 0, f'{max(WINDOWS)}_UD_{freq}_'] = pd.NA

        df[f'{max(WINDOWS)}_UD_targets_{freq}'] =  [df[f'{max(WINDOWS)}_UD_{freq}_'].iloc[max(0, i-target_window):i+1].dropna().tail(20).to_numpy() for i in range(len(df))]
        df[f'{max(WINDOWS)}_UD_targets_{freq}'] = df[f'{max(WINDOWS)}_UD_targets_{freq}'].apply(padding_zero)

        df[f'{max(WINDOWS)}_UD_{freq}'] =  [(np.array(df[f'{max(WINDOWS)}_UD_{freq}_'].iloc[max(0, i-target_window):i+1].dropna().tail(20).tolist() + [df['close'].iloc[i]],dtype=np.float32)- df['ref_px'].iloc[i])/NORM_FACTOR for i in range(len(df))]
        df[f'{max(WINDOWS)}_UD_{freq}'] = df[f'{max(WINDOWS)}_UD_{freq}'].apply(lambda x: padding_zero(x,length=21))

        
# df[f'{max(WINDOWS)}_U_targets_{freq}'] = [(df[f'{max(WINDOWS)}_U_{freq}'].iloc[max(0, i-target_window):i+1].dropna().tail(10).to_numpy() - df['close'].iloc[i])/norm_factor for i in range(len(df))]
# df[f'{max(WINDOWS)}_D_targets_{freq}'] = [(df[f'{max(WINDOWS)}_D_{freq}'].iloc[max(0, i-target_window):i+1].dropna().tail(10).to_numpy() - df['close'].iloc[i])/norm_factor for i in range(len(df))]

        df.drop(columns=[f'{max(WINDOWS)}_U_{freq}',f'{max(WINDOWS)}_D_{freq}',f'{max(WINDOWS)}_UD_{freq}_'], inplace=True)
    return df

def UD(df,window,freq):
    df = df.loc[~df[f'{window}_std_{freq}'].isna()]
    df.reset_index(inplace=True,drop=True)
    df = UD_cal(df,'close_1',window,freq)
    # df = flag_period_extreme(df, window,freq,source_col='U', new_col='U_peak', mode='max')
    # df = flag_period_extreme(df, window,freq,source_col='D', new_col='D_trough', mode='min')
    # df = to_norm_cols(df,freq,norm_cols=[f'{window}_U_{freq}',f'{window}_D_{freq}'])
    return df

def get_raw(df,window):
    df[f'raw_{window}'] = [df['close'].iloc[max(0, i-window):i+1].tolist() for i in range(len(df))]
    df[f'raw_{window}_high'] = [df['high'].iloc[max(0, i-window):i+1].tolist() for i in range(len(df))]
    df[f'raw_{window}_low'] = [df['low'].iloc[max(0, i-window):i+1].tolist() for i in range(len(df))]
    return df


def resampler(df): 
    if not df.empty:
        high_ind = df['high'].idxmax()
        low_ind =  df['low'].idxmin() 
        open = df['open'].iloc[0]
        close = df['close'].iloc[-1]
        res_df = {
                    'ts': df['ts'].iloc[0],
                    'open': open,
                    'high': df.at[high_ind, 'high'],
                    'low': df.at[low_ind, 'low'],
                    'close': close,
                    'volume': df['volume'].sum(),
                    'avg_px': ((df['avg_px']* df['volume']).sum()) / (df['volume'].sum()),
                    'rth':df['rth'].iloc[0],
                    'n_px': df['n_px'].iloc[-1],
                    'nts':df['nts'].iloc[0],
                    'is_news': np.stack(df['is_news'].values).max(axis=0),
                    'vix': df['vix'].iloc[-1]
                    }
        
        if high_ind == low_ind :
            res_df['hl'] = df.at[high_ind,'hl']
        elif high_ind < low_ind:
            res_df['hl'] = 1
        else:
            res_df['hl'] = 0
        
        if res_df['hl'] == 1:
            res_df['ohlc_nts'] = [res_df['nts'],df.at[high_ind, 'nts'],df.at[low_ind, 'nts'],df['nts'].iloc[-1]]
            res_df['ohlc_vec'] = [res_df['open'],res_df['high'],res_df['low'],res_df['close']]
        else:
            res_df['ohlc_nts'] = [res_df['nts'],df.at[low_ind, 'nts'],df.at[high_ind, 'nts'],df['nts'].iloc[-1]]
            res_df['ohlc_vec'] = [res_df['open'],res_df['low'],res_df['high'],res_df['close']]
            
        res_df['ohlc_day'] = [df['day'].iloc[0]]*4

        res_df['bar_range'] = res_df['high'] - res_df['low']
        res_df['bar_diff'] = res_df['close'] - res_df['open']

        return pd.Series(res_df) 

# def normalize_rows(arr):
#     """MinMax normalize each row independently"""
#     # Get min and max along axis=1 (per row)
#     row_min = arr.min()
#     row_max = arr.max()

#     range_val = row_max - row_min
  
#     normalized = (arr - row_min) / range_val
#     return normalized


def window_rollout(df,freq):

    if freq != '1':
        #ohlc_nts =   [np.concatenate(df['ohlc_nts'].iloc[max(0, i-window_size):i+1].tolist()) for i in range(len(df))]
        df['px_session'] = [np.concatenate(df['px_session'].iloc[max(0, i-WINDOW_SIZE):i+1].tolist())/NORM_FACTOR for i in range(len(df))] #NORM_FACTOR[freq]

        df['tmp_ref_px'] = [[x]*4 for x in df['ref_px']]
        
        df['ohlc_vec'] = list((np.concatenate(df['ohlc_vec'].iloc[max(0, i-WINDOW_SIZE):i+1].tolist())-np.concatenate(df['tmp_ref_px'].iloc[max(0, i-WINDOW_SIZE):i+1].tolist()))/NORM_FACTOR for i in range(len(df)))

        #df['ohlc_vec'] = df['ohlc_vec'].apply(lambda x: normalize_rows(x))

        # df['ohlc_day'] = [np.concatenate(df['ohlc_day'].iloc[max(0, i-WINDOW_SIZE):i+1].tolist()) for i in range(len(df))]
        # df['ohlc_day'] = df['ohlc_day'].apply(lambda x: (x-np.max(x))/ (np.max(x)-np.min(x)) if np.max(x)-np.min(x) != 0 else [0]*x.shape[0])
    else:
        df['px_session'] = [np.concatenate(df['px_session'].iloc[i:i+1].tolist())/NORM_FACTOR for i in range(len(df))] #NORM_FACTOR[freq]

        df['tmp_ref_px'] = [[x]*4 for x in df['ref_px']]
        
        df['ohlc_vec'] = list((np.concatenate(df['ohlc_vec'].iloc[i:i+1].tolist())-np.concatenate(df['tmp_ref_px'].iloc[i:i+1].tolist()))/NORM_FACTOR for i in range(len(df)))

    df['movement'] = [np.column_stack((vec,px_sess)) for vec,px_sess in zip(df['ohlc_vec'],df['px_session'])]

    df.drop(columns=['ohlc_nts','ohlc_vec','ohlc_day','px_session','tmp_ref_px'], inplace=True)
    return df


def window_rollout_1min(df,freq):
    # ohlc_nts =   [df['nts'].iloc[max(0, i-ONE_WINDOW_SIZE):i+1].to_numpy() for i in range(len(df))]
    # ohlc_vec = [df['close'].iloc[max(0, i-ONE_WINDOW_SIZE):i+1].to_numpy() for i in range(len(df))]
    # df['tem'] = [df['day'].iloc[max(0, i-ONE_WINDOW_SIZE):i+1].to_numpy() for i in range(len(df))]
    # df['tem'] = df['tem'].apply(lambda x: (x-np.max(x))/ (np.max(x)-np.min(x)) if np.max(x)-np.min(x) != 0 else [0]*x.shape[0])
    # px_session = [df['px_session'].iloc[max(0, i-ONE_WINDOW_SIZE):i+1].to_numpy() for i in range(len(df))]
    # df['movement'] = [np.column_stack((day,ts, vec,px_sess)) for day,ts, vec,px_sess in zip(df['tem'], ohlc_nts, ohlc_vec,px_session)]

    df['ohlc_vec'] = list(zip(df['open'],df['high'],df['low'],df['close']))
    
    mask = df['hl'] == 0
    cols = ['open','low','high','close']
    df.loc[mask, 'ohlc_vec'] = df.loc[mask, cols].apply(tuple, axis=1)

    df['ohlc_nts'] = [[x]*4 for x in df['nts']]
    df['ohlc_day'] = [[x]*4 for x in df['day']]
    df = window_rollout(df,freq)
    #one_min_movement = df.loc[:,['ts','movement']]
    #one_min_movement.to_pickle('one_min_movement.pkl')
    #np.save('one_min_movement.npy', one_min_movement)
    #df.drop(columns=['movement'], inplace=True)
    return df


def normalisation(df):
    norm_cols = ['avg_px','day_targets','bar_range','bar_diff','px_diff','20_std','20_atr']
    
    for col in ['pre_ohlc_targets','sma_targets']:
        df[col].ffill(inplace=True)

    targets_cols = [col for col in df.columns if 'target' in col]

    df['day_targets'] = [
                        np.concatenate(arrs,dtype=np.float32) 
                        for arrs  in zip(*[df[col] for col in targets_cols])
                        ]

    df.drop(columns=targets_cols, inplace=True)
    for freq in FREQS:
        norm_factor = NORM_FACTOR #2350#NORM_FACTOR[freq] 
        for col in norm_cols:
            if col == 'day_targets':
                if freq == '1': 
                    df[col] = (df[col]-df['close'])/norm_factor
            elif col in ['20_std','20_atr','bar_range','bar_diff','px_diff']:
                df[f'{col}_{freq}'] = df[f'{col}_{freq}']/norm_factor
            else:
                df[f'{col}_{freq}'] = (df[f'{col}_{freq}']-df['close'])/norm_factor
    
    return df


# def get_reward(df):
#     df = pd.DataFrame({'col': [1,2,3,3,5,6,7,8,9,10]})

#     # 反转列，使用rolling，再反转回来
#     values = df['col'].values
#     n = len(values)

#     df['max_idx_next_3'] = [
#         ( np.argmax(values[i:min(i+3, n)])) 
#         for i in range(n)
#     ]

#     # 添加一列显示最大值
#     df['max_value_next_3'] = [
#         values[i:min(i+3, n)].max() 
#         for i in range(n)
#     ]

def sigmoid_transform(x, threshold=3, steepness=2.5, cap=1):
    sign = np.sign(x)
    x = np.abs(x)
    # Sigmoid gives the S-shaped curve you want
    raw = 1 / (1 + np.exp(-steepness * (x - threshold)))
    
    # Scale and shift to start near 0 at x=1 and approach cap
    # Find value at x=1 to use as offset
    offset = 1 / (1 + np.exp(-steepness * (1 - threshold)))
    
    # Scale to ensure range is [0, cap]
    return cap * (raw - offset) / (1 - offset) * sign


def weighted_window_sum(x,weights):
    x = np.sum(x * weights)/np.sum(weights)
    return sigmoid_transform(x)

def get_inertia_growth(df,alpha = 0.61828):
    window = 20
    weights = alpha ** np.arange(window)[::-1]  
    df['inertia_growth'] = df['px_diff'].rolling(window=window).apply(lambda x: weighted_window_sum(x,weights))
    return df

def take_away_burnout_period(df):
    mask = df['20_UD_720'].apply(lambda x: np.all(np.asarray(x) != 0.0)) 
    first_index = df.index[mask].min() if mask.any() else None
    df = df.iloc[first_index:]
    df.reset_index(drop=True, inplace=True)
    return df

def to_stack(df,freq,col_name,stack_cols=None,drop_cols=True):
    all_values = df[stack_cols].values  # shape: (len(df), len(stack_cols))
    df[f'{col_name}_{freq}'] = [np.array(all_values[idx_list],dtype=np.float32) if isinstance(idx_list, list) else pd.NA for idx_list in df[f'index_{freq}']]
    if drop_cols:
        df.drop(columns=stack_cols, inplace=True)
    return df

def get_stack_cols(extra_cols,stack_cols):
    for col in extra_cols:
        for freq in FREQS:
            if freq != '1':
                stack_cols += [f'{col}_{freq}']
    return stack_cols

def get_skip_night(df):
    df.reset_index(inplace=True)

    tem = df[df['rth']==0].groupby('sess').agg(

        first_index=('index', 'first'),
        last_index=('index', 'last'),

        # Add more with custom names
    )
    tem.reset_index(inplace=True)
    tem['first_index'] = tem['first_index'] -1
    tem['first_index'] = tem['first_index'].astype(int).shift(-1)

    df['skip_night'] = pd.NA

    df.loc[tem['last_index'].tolist(),'skip_night'] = tem['first_index'].tolist()
    return df


def feature_preprocessing(df):
    
    to_stack_dict ={
                     'directional': ['20_UD_flag','20_fvg'],
                     'px_session': ['bar_range','bar_diff','bar_score','px_diff'],
                            'vol': ['20_std','20_atr','volume','vix','fomc','nfp','cpi','ppi','gdp'],
                            'tas': ['inertia_growth','20_sma_rsi','20_bbands','avg_px','needle','large_candle']
                
    }
    
    
    is_news_NAMES = ['fomc', 'nfp', 'cpi', 'ppi', 'gdp']
    for freq in FREQS:
        for i, name in enumerate(is_news_NAMES):
            df[f'{name}_{freq}'] = df[f'is_news_{freq}'].apply(lambda x: x[i] if isinstance(x, np.ndarray) else np.nan)
        df.drop(columns=[f'is_news_{freq}'], inplace=True)

    for freq in FREQS:
        for col_name, stack_cols in to_stack_dict.items():
            stack_cols = [ f'{col}_{freq}' for col in stack_cols ]
            if freq != '1':
                if col_name != 'pivots':
                    df = to_stack(df,freq, col_name,stack_cols= stack_cols)
            else:
                if col_name == 'tas':
                    extra_cols = ['20_sma_rsi','20_bbands']
                    stack_cols = get_stack_cols(extra_cols,stack_cols)
                    
                elif col_name == 'vol':
                    pass  # keep only 9 base cols, no cross-freq stacking
 
                
                df[f'{col_name}_{freq}'] = list(df[stack_cols].to_numpy(copy=False).astype(np.float32))

                drop_cols = [col for col in stack_cols if col.endswith('_1')]

                df.drop(columns=drop_cols, inplace=True)

        if freq != '1':
            df.drop(columns=[f'index_{freq}'], inplace=True)    

    df.drop(columns=['ref_px'], inplace=True)

    first_index = df['tas_720'].dropna().index[max(WINDOWS)-1] 
    df = df.iloc[first_index:]
    df.reset_index(drop=True, inplace=True)
    return df 

class dataPreprocessing:

    def __init__(self) -> None:
        self.dfs_dict = {}

    def get_data(self,force_update=False):
        if os.path.exists(f'./data/trading_data.pkl') and not force_update:
            df = pd.read_pickle(f'./data/trading_data.pkl')
            
            # train_data = pd.read_pickle(f'./data/train_data.pkl')
            # val_data = pd.read_pickle(f'./data/val_data.pkl')
            # test_data = pd.read_pickle(f'./data/test_data.pkl')
        else:
            self.prepare_data()
            
            df = self.agg_df()
         
            df = self.exclude_burnout_period(df) 
            df = normalisation(df)
            df = self.get_index(df) 
            df = feature_preprocessing(df)
            df = take_away_burnout_period(df)
            #df = self.get_the_final_targets(df)
            df = self.split_data(df)

            
        return df

    def prepare_data(self):
        for freq in FREQS:
            
            if freq == '1':
                df = np.load(DATA_PATH+'/data_reward.pkl', allow_pickle=True)

                df['avg_px'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

                open_high = (df['open'] - df['high']).abs()
                open_low = (df['open'] - df['low']).abs()
                close_high = (df['close'] - df['high']).abs()
                close_low = (df['close'] - df['low']).abs()
                df['hl'] = np.where(open_high < open_low, 1,
                           np.where(open_high > open_low, 0,
                           np.where(close_low <= close_high, 1, 0)))
                df['vix'] = normalize_vix_ewma(df['vix'])
                df['n_px']= (df['close'] - 40000)/40000 -1
                df = df[df['ts']>=BEGIN_DATE]
                df = df.drop_duplicates(subset=['ts'])
                df.reset_index(drop=True,inplace=True)
                df.loc[df['hl']==-1,'hl'] = 0
                df['bar_range'] = df['high'] - df['low']
                df['bar_diff'] = df['close'] - df['open']
                df = get_ref_px(df)
                df = get_high_low_targets(df)
                df.reset_index(drop=True,inplace=True)
                df.reset_index(inplace=True)
                df_ref = df.copy()
                # df = df.groupby(pd.Grouper(freq='120min', key='tem_ts')).agg({
                #                                                         'open': 'first',   
                #                                                         'high': 'max',      
                #                                                         'low': 'min',       
                #                                                         'close': 'last'    
                #                                                     })
                # df.to_excel('./120min_ohlcv.xlsx')
                # exit()
            else:
                
                if os.path.exists(f'./{freq}_ohlcv.pkl') and False:
                    df = pd.read_pickle(f'./{freq}_ohlcv.pkl')
                    pass
                    #df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','avg_px','rth','nts','hl','ohlc_ts','ohlc_vec','bar_range','bar_diff'])
                
                else:
                    df = df_ref.copy()
                    freq_ = 'W-MON' if freq == 'W' else f'{freq}min'
                    df = df.groupby(pd.Grouper(freq=freq_,key='ts'),group_keys=False).apply(resampler)
                    df.reset_index(drop=True, inplace=True)
                    df = df[df['ts'].notna()]
                    df.reset_index(drop=True, inplace=True)
                    if freq !='720':
                        df = get_ref_px(df)
                    else:
                        df['ref_px'] = df['open'].shift(-1).ffill()
                    df.to_pickle(f'./data/{freq}_ohlcv.pkl')
                    df.to_excel(f'./data/sample/{freq}_ohlcv.xlsx')

                df = df[df['ts']>=BEGIN_DATE]
                
                df.reset_index(drop=True,inplace=True)
                for window in WINDOWS:
                    df = get_raw(df,window)
            
            df = preprocessing(df,freq)
            df['px_diff'] = df['close'] - df['close'].shift(1)
            df = get_inertia_growth(df)

            for window in WINDOWS:
                for feature in FEATURES:
                    target_func = globals()[feature]
                    df = target_func(df,window)

            df = get_needle(df)
            df = get_candle(df)
            
            if freq != '1':
                df = window_rollout(df,freq)
                df.drop(columns=['nts','rth','ref_px'], inplace=True)
            else:
                df = window_rollout_1min(df,freq)

            target_cols = list(set(list(df.columns)) - set(['ts','nts','sess','rth','session_end','day','ref_px','pre_ohlc_targets','sma_targets','intrady_day_hl_targets','original_signal',	'reward',	'signal']))

            df.rename(columns={col: f'{col}_{freq}' for col in target_cols}, inplace=True)
            #df=df.set_index(['ts']).add_suffix(f'_{freq}').reset_index()

            self.dfs_dict[freq] = df
        

    def agg_df(self):
        for freq, tem_df in self.dfs_dict.items():
            
            if freq == '1':
                df = tem_df
                ref_ts = set(tem_df['ts'].unique())
                continue 
                
            
            tem_df_cols = list(tem_df.columns)
            
            tem_df_cols.remove('ts')
            #tem_df_cols = [f'20_std_{freq}']

            tem_ts = set(tem_df['ts'].unique())
            missing_ts = tem_ts - ref_ts
            if missing_ts: 
                print(freq,missing_ts)
                exit()
            
            df = df.merge(tem_df, on='ts', how='outer')
            
            #debug_df = df[df['open_1'].isna()]

            target_index = df[df[tem_df_cols].notna().all(axis=1)].index
            target_index = [ind-1 for ind in target_index][2:]     

            df[tem_df_cols] = df[tem_df_cols].ffill()
            
            df.loc[~df.index.isin(target_index),tem_df_cols]= pd.NA

        for col in df.columns:
            if 'raw' in col:
                #df[col] = df[col].shift(1)
                df[col].ffill(inplace=True)

        for freq in FREQS:
            if freq == '1':
                window = max(WINDOWS)
                df = UD(df,window,freq)
                continue
            for unit_ta in UNIT_TAS:
                # if 'price_session' == unit_ta:
                #     df = uni_price_session(df,freq)
                
                if 'UD' == unit_ta:
                    window = max(WINDOWS)
                    df = UD(df,window,freq)
                else:    
                    for window in WINDOWS:
                        target_func = globals()[f'unit_{unit_ta}']
                        if unit_ta == 'atr':
                            df[f'{window}_atr_{freq}'] = df.apply(
                                                                    lambda x: unit_atr(
                                                                        x[f'raw_{window}_high_{freq}'] + [x['high_1']],
                                                                        x[f'raw_{window}_low_{freq}'] + [x['low_1']],
                                                                        x[f'raw_{window}_{freq}'] + [x['close_1']],
                                                                        window
                                                                    ) if isinstance(x[f'raw_{window}_{freq}'], list) else None,
                                                                    axis=1
                                                                )
                        else:
                            df[f'{window}_{unit_ta}_{freq}'] = df.apply(
                                                                        lambda x: target_func(x[f'raw_{window}_{freq}'] + [x['close_1']], window)
                                                                        if isinstance(x[f'raw_{window}_{freq}'], list) else None,
                                                                        axis=1
                                                                        )

        drop_cols = [col for col in df.columns if 'raw' in col or ('hl_' in col and col != 'hl_1')]# + ['rth'] #del open_15...., rename open_1
        
        drop_cols.remove('intrady_day_hl_targets')

        ohlc_cols = ['open','high','low','close','hl']
        for freq in set(FREQS):
            if freq == '1':
                df.rename(columns={f'{col}_1': col for col in ohlc_cols}, inplace=True)
            else:
                drop_cols += [f'{col}_{freq}' for col in ohlc_cols]

        df.drop(columns=drop_cols, inplace=True)

        # for freq in FREQS:
        #     for feature in ['20_std','bar_range','px_diff','bar_diff','bar_score','n_volume']:
        #         print(freq,feature,df[f'{feature}_{freq}'].mean(),df[f'{feature}_{freq}'].std())
  
        def ewma_normalize(df, feature, freq):
            optimal_spans = {
                '1': 40000,   # 1-min: Larger span for more smoothing (≈7 days half-life)
                '15': 8000,   # 15-min: (≈17 days half-life)
                '30': 6000,   # 30-min: (≈17 days half-life)
                '60': 4000,   # 60-min: Your current span=1000 is OK (≈14 days half-life)
                '720': 1000,    # 12-hour: Smaller span for faster adaptation (≈50 days half-life)
                'W': 200
            }
            """Intelligent EWMA normalization with frequency-optimized spans"""
            col = f'{feature}_{freq}'
            span = optimal_spans.get(freq, 1000)
            
            # Calculate EWMA statistics
            ewma_mean = df[col].ewm(span=span).mean()
            ewma_std = df[col].ewm(span=span).std()
            
            # Normalize
            normalized = (df[col] - ewma_mean) / ewma_std.clip(lower=1e-8)
            
            return normalized
        
        for freq in FREQS:
            for feature in ['volume']:
                df[f'{feature}_{freq}'] = ewma_normalize(df, feature, freq)
            
            df[f'20_sma_rsi_{freq}'] = df[f'20_sma_rsi_{freq}'].apply(transform_rsi_data)
            #df[f'20_sma_{freq}'] = df[f'20_sma_{freq}']/OHLC_NORM_FACTOR[freq]

        df = get_UD_targets(df)

        # for year in df['ts'].dt.year.unique():
        #     print(f"\n=== Year {year} ===")
        #     year_df = df[df['ts'].dt.year == year]
        #     for freq in FREQS:
        #         for feature in ['20_std','bar_range','px_diff','bar_diff','bar_score','n_volume']:
        #             col = f'{feature}_{freq}'

        #             if col in year_df.columns:
        #                 print(f"{freq} {feature}: μ={year_df[col].mean():.4f}, σ={year_df[col].std():.4f}")            
        return df


    def get_the_final_targets(self,df):
        target_cols = [col for col in df.columns if 'target' in col]

        for col in target_cols:
            df[col].ffill(inplace=True)

        t1 = time.time()
        def get_nearest(arr,pos_neg ='pos',mode='nearest',threshold=100):
    
            if pos_neg == 'pos':
                pos_neg_mask = (arr > 0)
            else:
                pos_neg_mask = (arr < 0)
            mask = pos_neg_mask & (np.abs(arr) < threshold) 
            candidates = arr[mask]
            
            if len(candidates) == 0:
                return 0
            
            if mode == 'threshold':
                distances = np.abs(np.abs(candidates) - threshold)
                res = candidates[np.argmin(distances)]
            elif mode =='nearest':
                distances = np.abs(candidates)
                res = candidates[np.argmin(distances)]

            return res

        # df['targets'] = [
        #                 np.sort(np.concatenate(arrs))
        #                 for arrs in zip(*[df[col] for col in target_cols])
        #                 ]

        for target_col in target_cols:
            df[target_col] = df[target_col].apply(lambda x: np.sort(x))
            #for mode in ['threshold','nearest']:
            
            df[f'neg_{target_col}'] = df[target_col].apply(get_nearest,pos_neg='neg')
            df[f'pos_{target_col}'] = df[target_col].apply(get_nearest,pos_neg='pos')
        
        #df.drop(columns=target_cols, inplace=True)
        print('time taken:',time.time() - t1)
        return df

    def exclude_burnout_period(self,df):
        tem_indices = df[df['sma_targets'].notnull()].index
        target_index = next(
                            (idx for idx in tem_indices 
                            if not np.isnan(df.loc[idx, 'sma_targets']).any()),
                            None
                            )
        df = df.iloc[target_index:-1]
        df.reset_index(drop=True, inplace=True)
        return df

    def get_index(self,df):
        windows_mapping = {
                            '15': 600,
                            '30': 1200,
                            '60': 1200*2,
                            '720': int(14400*1.5),
                            'W': 72000
                          }
        
        for freq in FREQS:
            if freq != '1':
                target_window = windows_mapping[freq]
                df[f'index_{freq}'] = [df[f'movement_{freq}'].iloc[max(0, i-target_window):i+1].dropna().tail(WINDOW_SIZE).index.tolist() if isinstance(df[f'movement_{freq}'].iloc[i], np.ndarray) else pd.NA for i in range(len(df))]
        return df

    def create_chunk_info(self,df):
        if 'index' not in df.columns:
            df = df.reset_index()
        tem = df.groupby('sess').agg(
                                    count=('index', 'count'),
                                    first_index=('index', 'first'),
                                    last_index=('index', 'last'),
                                    )
        tem.reset_index(inplace=True)
        tem.to_pickle('./chunk_info.pkl')


    # def create_is_valid_mask(self,df,last_n=10,next_n=5):
    #     df['is_valid'] = True

    #     # Create a boolean mask
    #     mask = pd.Series(False, index=df.index)

    #     for idx in df.index[df['session_end']]:
    #         start = max(0, idx - (last_n-1))
    #         end = min(len(df) - 1, idx + next_n)
    #         mask.iloc[start:end+1] = True

    #     df.loc[mask, 'is_valid'] = False
    #     return df

    def split_data(self,df):
        
        # first_index = df[df['ts'].dt.time.eq(pd.Timestamp('02:59:00').time())].index[0]
        # df = df.iloc[first_index:]
        # df.reset_index(drop=True, inplace=True)

        # df.reset_index(drop=True, inplace=True)

        df['ts']= df['ts'].astype(str)
        df = df.astype('float32', errors='ignore')
       # df = self.create_is_valid_mask(df)
     #   df = get_skip_night(df)
        
        self.create_chunk_info(df)
        
        df.to_pickle(f'./data/trading_data.pkl')
        df.head(10000).to_excel(f'./data/sample/trading_data_sample.xlsx',index=False)

        return df
    




# def correlation_analysis():
#     data = {'Column_A': [1, 2, 3, 4, 5],
#             'Column_B': [2, 4, 5, 4, 5]}

#     df = pd.DataFrame(data)

#     # Calculate correlation between two columns
#     correlation = df['Column_A'].corr(df['Column_B'])

#     print(f"Correlation coefficient: {correlation:.3f}")


if __name__ == "__main__":

    dataPreprocessing().get_data(force_update=True)













