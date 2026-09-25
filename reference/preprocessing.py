import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import List, Optional
import numpy as np
import datetime
import time
import params
from collections import deque


def _compute_one_variant(args):
    """Worker: compute one strategy variant. Returns (name, array of [unr, ls, ewm] per bar)."""
    sname, sfn_name, sparams, df_pkl_path, round_trip, ewm_span, ewm_alpha, dpp = args

    # Re-import inside worker process
    if sfn_name == 's4':
        from strategies.strategy_s4_livermore import generate_signals as sfn
    else:
        from strategies.strategy_s9_combined import generate_signals as sfn

    df = pd.read_pickle(df_pkl_path)
    signals = sfn(df, sparams)
    closes = df['close'].values
    n = len(signals)

    unr_arr = np.zeros(n, dtype=np.float32)
    ls_arr = np.zeros(n, dtype=np.float32)
    ewm_arr = np.zeros(n, dtype=np.float64)
    wr_arr = np.zeros(n, dtype=np.float32)
    rpnl_arr = np.zeros(n, dtype=np.float32)

    position = 0
    entry_close = 0.0
    closed_pnls = []

    signal_at_close = np.zeros(n, dtype=np.float32)
    for j in range(1, n):
        signal_at_close[j - 1] = signals[j]

    for j in range(n):
        c = closes[j]
        target = int(signal_at_close[j])

        if position != target:
            if position != 0:
                pnl = (c - entry_close) * position * dpp
                closed_pnls.append(pnl - round_trip)
            if target != 0:
                position = target
                entry_close = c
            else:
                position = 0

        if position != 0:
            unrealised = (c - entry_close) * position * dpp
            unr_arr[j] = (unrealised - round_trip) / 320.0
            ls_arr[j] = float(position)

        n_trades = len(closed_pnls)
        if n_trades >= 2:
            trades = closed_pnls[-ewm_span:]
            nt = len(trades)
            w = [(1 - ewm_alpha) ** (nt - 1 - k_) for k_ in range(nt)]
            w_sum = sum(w)
            mu = sum(wj * pj for wj, pj in zip(w, trades)) / w_sum
            var = sum(wj * (pj - mu) ** 2 for wj, pj in zip(w, trades)) / w_sum
            sigma = var ** 0.5
            raw_sharpe = mu / sigma if sigma > 1e-8 else 0.0
            confidence = min(n_trades / 30.0, 1.0)
            ewm_arr[j] = raw_sharpe * confidence

            # EWM-weighted win rate
            wins = [1.0 if p > 0 else 0.0 for p in trades]
            wr_arr[j] = sum(wj * wn for wj, wn in zip(w, wins)) / w_sum * confidence

            # EWM-weighted aggregate realised PnL (normalised by 320)
            rpnl_arr[j] = mu / 320.0 * confidence

    result = np.column_stack([unr_arr, ls_arr, ewm_arr, wr_arr, rpnl_arr]).astype(np.float32)
    return sname, [result[j] for j in range(n)]


def compute_strat_columns(df, strat_configs, round_trip, ewm_span, ewm_alpha, dpp, n_workers=None):
    """Compute all strategy variants. Uses multiprocessing if n_workers > 1."""
    import multiprocessing as mp
    import tempfile

    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(strat_configs))

    n_60 = len(df)

    if n_workers <= 1:
        # Serial fallback
        closes_60 = df['close'].values
        for vi, (sname, sfn, sparams) in enumerate(strat_configs):
            signals = sfn(df, sparams)
            unr_arr = np.zeros(n_60, dtype=np.float32)
            ls_arr = np.zeros(n_60, dtype=np.float32)
            ewm_arr = np.zeros(n_60, dtype=np.float64)
            position = 0; entry_close = 0.0; closed_pnls = []
            signal_at_close = np.zeros(n_60, dtype=np.float32)
            for j in range(1, n_60):
                signal_at_close[j - 1] = signals[j]
            for j in range(n_60):
                c = closes_60[j]; target = int(signal_at_close[j])
                if position != target:
                    if position != 0:
                        pnl = (c - entry_close) * position * dpp
                        closed_pnls.append(pnl - round_trip)
                    if target != 0: position = target; entry_close = c
                    else: position = 0
                if position != 0:
                    unr_arr[j] = ((c - entry_close) * position * dpp - round_trip) / 320.0
                    ls_arr[j] = float(position)
                n_trades = len(closed_pnls)
                if n_trades >= 2:
                    trades = closed_pnls[-ewm_span:]
                    nt = len(trades)
                    w = [(1 - ewm_alpha) ** (nt - 1 - k_) for k_ in range(nt)]
                    w_sum = sum(w)
                    mu = sum(wj * pj for wj, pj in zip(w, trades)) / w_sum
                    var = sum(wj * (pj - mu) ** 2 for wj, pj in zip(w, trades)) / w_sum
                    sigma = var ** 0.5
                    raw_sharpe = mu / sigma if sigma > 1e-8 else 0.0
                    ewm_arr[j] = raw_sharpe * min(n_trades / 30.0, 1.0)
            df[sname] = [np.array([u, ls, ew], dtype=np.float32)
                         for u, ls, ew in zip(unr_arr, ls_arr, ewm_arr)]
            if (vi + 1) % 50 == 0 or vi == len(strat_configs) - 1:
                print(f'    {vi + 1}/{len(strat_configs)} variants done')
        return df

    # Parallel: save df to temp pickle, workers reload it
    tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    tmp_path = tmp.name
    tmp.close()
    df.to_pickle(tmp_path)

    # Build args — use string for function name (can't pickle functions)
    worker_args = []
    for sname, sfn, sparams in strat_configs:
        fn_name = 's4' if 'S4' in sname else 's9'
        worker_args.append((sname, fn_name, sparams, tmp_path,
                           round_trip, ewm_span, ewm_alpha, dpp))

    print(f'    Launching {n_workers} workers for {len(strat_configs)} variants...')
    with mp.Pool(n_workers) as pool:
        results = pool.map(_compute_one_variant, worker_args)

    os.unlink(tmp_path)

    # Concat all at once to avoid fragmentation
    new_cols = {sname: col_data for sname, col_data in results}
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    print(f'    {len(results)} variants collected')

    return df




def unit_std(px_list,window):
    px_list = px_list[-window:]
    return np.std(px_list)

def unit_avedev(px_list,window):
    px_list = px_list[-window:]
    mean = np.mean(px_list)
    return np.mean(np.abs(np.array(px_list) - mean))

# n_close, close
def std(df,window,px_col='close'):
    df[f'{window}_std'] = df[px_col].rolling(window=window).std(ddof=1)
    return df

def avedev(df,window,px_col='close'):
    df[f'{window}_avedev'] = df[px_col].rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return df



def atr(df: pd.DataFrame, window: int = 14, 
        high_col: str = 'high', low_col: str = 'low', close_col: str = 'close',
        ) -> pd.DataFrame:
    """
    Vectorized Wilder's Average True Range (ATR)
    Adds ATR column to the DataFrame (or returns Series if output_col=None)
    
    Similar style to your sma_rsi function.
    """
    period = window 
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
    n = window 
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
                    }
        

        return pd.Series(res_df) 



# 




def get_raw(df,window):
    df[f'raw_{window}'] = [df['close'].iloc[max(0, i-window):i+1].tolist() for i in range(len(df))]
    df[f'raw_{window}_high'] = [df['high'].iloc[max(0, i-window):i+1].tolist() for i in range(len(df))]
    df[f'raw_{window}_low'] = [df['low'].iloc[max(0, i-window):i+1].tolist() for i in range(len(df))]
    return df



class dataPreprocessing:

    def __init__(self) -> None:
        self.dfs_dict = {}

    def get_data(self,force_update=False):
        if os.path.exists(f'../data/trading_data.pkl') and not force_update:
            df = pd.read_pickle(f'../data/trading_data.pkl')
            # train_data = pd.read_pickle(f'./data/train_data.pkl')
            # val_data = pd.read_pickle(f'./data/val_data.pkl')
            # test_data = pd.read_pickle(f'./data/test_data.pkl')
        else:
            self.prepare_data()
            df = self.agg_df()
            df = self.exclude_burnout_period(df) 
            df = self.split_data(df)
        return df

    def prepare_data(self):
        for freq in params.FREQS:
            
            if freq == '1':
                df = pd.read_pickle(params.DATA_PATH+'/data.pkl')
                df = df[df['ts']>=params.BEGIN_DATE]
                df = df.drop_duplicates(subset=['ts'])
                df.reset_index(drop=True,inplace=True)
                df.reset_index(inplace=True)
                df_ref = df.copy()

            else:
                
                if os.path.exists(f'./{freq}_ohlcv.pkl') and False:
                    pass
                else:
                    df = df_ref.copy()
                    freq_ = 'W-MON' if freq == 'W' else f'{freq}min'
                    df = df.groupby(pd.Grouper(freq=freq_,key='ts'),group_keys=False).apply(resampler)
                    df.reset_index(drop=True, inplace=True)
                    df = df[df['ts'].notna()]
                    df.reset_index(drop=True, inplace=True)
                    df.to_pickle(f'../data/{freq}_ohlcv.pkl')
                    df.to_excel(f'../data/sample/{freq}_ohlcv.xlsx')


                df = df[df['ts']>=params.BEGIN_DATE]
                
                if freq == '60':
                    from strategies.backtest_engine import DOLLAR_PER_POINT, COMM_PER_TRADE

                    strat_configs = params.build_strat_configs()
                    print(f'  Computing {len(strat_configs)} strategy variants...')

                    df = compute_strat_columns(
                        df, strat_configs,
                        round_trip=COMM_PER_TRADE * 2,
                        ewm_span=100,
                        ewm_alpha=2.0 / 201,
                        dpp=DOLLAR_PER_POINT,
                        n_workers=12)

                    print(f'  Added {len(strat_configs)} strategy columns to 60min')

                df.reset_index(drop=True,inplace=True)
                for window in params.WINDOWS:
                    df = get_raw(df,window)
            

            for window in params.WINDOWS:
                for feature in params.FEATURES:
                    if freq == '1':
                        target_func = globals()[feature]
                        df = target_func(df,window)
            
            

            target_cols = list(set(list(df.columns)) - set(['ts','nts','sess','rth','session_end','day','ref_px','pre_ohlc_targets','sma_targets','intrady_day_hl_targets','original_signal',	'reward',	'signal']))

            df.rename(columns={col: f'{col}_{freq}' for col in target_cols}, inplace=True)

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
                df[col].ffill(inplace=True)


        df = df.copy()  # defragment before unit_TA loop

        for freq in params.FREQS:
            if freq == '1':
                continue  # no raw_20 data for 1min, skip unit_TAs
            for unit_ta in params.UNIT_TAS:
                for window in params.WINDOWS:
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

        

        drop_cols = [col for col in df.columns if 'raw' in col ]# + ['rth'] #del open_15...., rename open_1
        
        ohlc_cols = ['open','high','low','close']
        for freq in set(params.FREQS):
            if freq == '1':
                df.rename(columns={f'{col}_1': col for col in ohlc_cols}, inplace=True)
            else:
                drop_cols += [f'{col}_{freq}' for col in ohlc_cols]

        df.drop(columns=drop_cols, inplace=True)

        return df


    def exclude_burnout_period(self,df):
        first_ind = df[df['20_atr_60'].notnull()].index[0]

        df = df.iloc[first_ind:-1]
        df.reset_index(drop=True, inplace=True)
        return df

    def get_index(self,df):
        params.WINDOWS_mapping = {
                            '15': 600,
                            '30': 1200,
                            '60': 1200*2,
                            '720': int(14400*1.5),
                            'W': 72000
                          }
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


    def split_data(self,df):

        df['ts']= df['ts'].astype(str)
        df = df.astype('float32', errors='ignore')
        
        self.create_chunk_info(df)
        
        df.to_pickle('../data/trading_data.pkl')
        df.head(10000).to_excel('../data/sample/trading_data_sample.xlsx',index=False)

        return df
    


if __name__ == "__main__":

    dataPreprocessing().get_data(force_update=True)


