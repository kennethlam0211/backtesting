import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import datetime
from params import NORM_FACTOR



def get_reward(df,freq):


    def get_min_max(df,window=15):
        window = 60
        values = df['close'].values
        n = len(values)

        # df['max_idx'] = [
        #     ( np.argmax(values[i:min(i+5, n)])) 
        #     for i in range(n)
        # ]
        # df['min_idx'] = [
        #     ( np.argmin(values[i:min(i+5, n)])) 
        #     for i in range(n)
        # ]
        
        df['max'] = [
                        values[i:min(i+window, n)].max() 
                        for i in range(n)
                    ]
        
        df['min'] = [
                        values[i:min(i+window, n)].min() 
                        for i in range(n)
                    ]
        return df


    expected_reward = 100

    stop_reward = 0

    df['ts'] = pd.to_datetime(df['ts'])

    df['tem_ts'] = df['ts'].apply(lambda x: x - datetime.timedelta(hours=5))
    
    df = df.groupby(pd.Grouper(freq='720min', key='tem_ts')).apply(get_min_max,window=int(freq))

    df['long_reward'] = df['max'] - df['open']

    df['short_reward'] = df['open'] - df['min']

    df['temp_reward'] = df['long_reward'] - df['short_reward']

    df['open_diff']=df['open']-df['open'].shift(-1)

    def calculate_direction_numpy(df, stop_reward):
        temp = df['temp_reward'].values
        # direction = np.zeros(len(df), dtype=int)
        direction = df['direction'].to_numpy()
        day_night = df['day_night'].to_numpy()
        
        for i in range(len(temp)):
            # if temp[i] >= expected_reward:
            #     direction[i] = 1
            # elif temp[i] <= -expected_reward:
            #     direction[i] = -1
            if day_night[i] != day_night[i-1]:
                direction[i] = 0
            elif direction[i-1] == -1 and temp[i] <= stop_reward:
                direction[i] = -1
            elif direction[i-1] == 1 and temp[i] >= stop_reward:
                direction[i] = 1
        
        return direction


    # df.loc[(df['direction'].shift(1)==1)&(df['temp_reward']>stop_reward) , 'direction'] = 1
    # df.loc[(df['direction'].shift(1)==-1)&(df['temp_reward']<-stop_reward) , 'direction'] = -1

    df['direction'] = 0
    df.loc[df['temp_reward'] > expected_reward, 'direction'] = 1
    df.loc[df['temp_reward'] < -expected_reward, 'direction'] = -1
    df.loc[df['day_night']!=df['day_night'].shift(-1),'direction'] = 0


    df['direction'] = calculate_direction_numpy(df, stop_reward)

    df['reward'] = 0
    df.loc[df['direction'] == 1, 'reward'] = df['long_reward']
    df.loc[df['direction'] == -1, 'reward'] = df['short_reward']


    # df['risk'] = 0
    # df.loc[df['direction'] == 1, 'risk'] = df['short_reward']
    # df.loc[df['direction'] == -1, 'risk'] = df['long_reward']

    def get_is_range(gp):
        if not gp.empty:
            temp_long_reward= gp['open_diff'].sum()*-1
            temp_short_reward = -temp_long_reward
            B = gp['open'].iloc[0]
            C = gp['close'].max()
            D = gp['close'].min()
            flat_reward = np.maximum(C - B, B - D)- np.minimum(C - B, B - D) * 2 + np.abs(temp_short_reward)
            #gp.drop(columns=['temp_long_reward','temp_short_reward'], inplace=True)
            return pd.Series({'flat_reward': flat_reward})

    def calculate_holding_periods(df):
        # Create groups of consecutive same values
        df['group'] = (df['direction'] != df['direction'].shift(1)).cumsum()
        
        tem = df[df['direction']==0].groupby(by=['group','day_night']).apply(get_is_range)
        print('flat reward for 0:',tem['flat_reward'].mean())
        #tem.to_excel('0.xlsx')
        tem = df[df['direction'].isin([1,-1])].groupby(by=['group','day_night']).apply(get_is_range)
        print('flat reward for 1 and -1:',tem['flat_reward'].mean())
        #tem.to_excel('1-1.xlsx')

        # Calculate holding periods
        holding_periods = df.groupby('group')['direction'].agg(['first', 'count'])
        
        # Separate by direction
        holding_1 = holding_periods[holding_periods['first'] == 1]['count']
        holding_minus1 = holding_periods[holding_periods['first'] == -1]['count']
        
        print(f"hp1: {holding_1.mean():.2f}")
        print(f"hp-1: {holding_minus1.mean():.2f}")

        reward = df.groupby(by=['group','direction']).agg({'open_diff':'sum'})

        reward = reward.reset_index()

        count_1 = reward[reward['direction'] == 1]['group'].count()
        count_minus1 = reward[reward['direction'] == -1]['group'].count()
        print('count:',count_1,count_minus1)

        reward.loc[reward['direction'] == 1, 'open_diff'] = reward['open_diff']*-1

        reward_1 = reward[reward['direction'] == 1]['open_diff'].mean().round(0)
        reward_minus1 = reward[reward['direction'] == -1]['open_diff'].mean().round(0)
        print('mean reward:',reward_1,reward_minus1)
        
        
        reward_1 = reward[reward['direction'] == 1]['open_diff'].sum()
        reward_minus1 = reward[reward['direction'] == -1]['open_diff'].sum()
        print('agg reward:',reward_1,reward_minus1)
        
        df['reward'] = df['reward']/NORM_FACTOR
        # risk = df.groupby(by=['group','direction']).agg({'risk':'sum'})
        
        # risk = risk.reset_index()
        # risk_1 = risk[risk['direction'] == 1]['risk'].sum()
        # risk_minus1 = risk[risk['direction'] == -1]['risk'].sum()
        # print('risk:',risk_1,risk_minus1)

    # Usage
    calculate_holding_periods(df)


    df = df.reset_index(drop=True)

    df.drop(columns=['long_reward','short_reward','temp_reward','group','open_diff','tem_ts','min','max'], inplace=True)

    return df

if __name__ == "__main__":
    DATA_PATH = r'C:\Users\kennethlam\Desktop\small_projects\data_cleaning\HSI_training_data'
    data = np.load(DATA_PATH+f'/1_ohlcv.npy', allow_pickle=True)
    df = pd.DataFrame(data, columns=['index','ts','open','open_1','open_2','high','low','close','volume','avg_px','hl','day_night','close_10','hour'])
    df = df[~( (df['ts'] >= datetime.datetime(2020, 3, 12, 12, 37, 0))  & (df['ts'] < datetime.datetime(2020, 3, 13, 1, 0, 0) ) )]
    df['day_night'] = np.where(df['day_night'] == 0, 1,np.where(df['day_night'] == 1, 0, df['day_night']))
    df = get_reward(df,'60')
    value_counts = df['direction'].value_counts()
    print(value_counts)
    #df.to_excel('1_ohlcv_reward.xlsx',index=False)













