
import numpy as np
import pandas as pd
import operator

col_names = [
    'start_ind','ts','price','volume',
    'high_1', 'low_1','next_ind_1',
    'high_2', 'low_2','next_ind_2',
    'high_2-', 'low_2-','next_ind_2-',
    'high_3', 'low_3','next_ind_3',
    'high_3-', 'low_3-','next_ind_3-',
    'high_4', 'low_4','next_ind_4',
    'high_4-', 'low_4-','next_ind_4-',
    'high_5', 'low_5','next_ind_5', 
    'high_10', 'low_10','next_ind_10',
    'high_15', 'low_15','next_ind_15',
    'high_20', 'low_20','next_ind_20',
    'high_30', 'low_30', 'next_ind_30',
    'high_60', 'low_60','next_ind_60',
    'high_720', 'low_720','next_ind_720'
    ]


col_to_ind_dict = {col_name:index for index, col_name in enumerate(col_names)}

next_freq_ind_dict = {"720":"60","60":"30","30":"15","15":"5","20":"10","10":"5","5":"1","2-":"1","2":"1","3":"1","4":"2","3-":"1","4-":"1"}

def search_(data,begin_ind,end_ind,target_high,target_low):   #ref_ind for method2 to keep the smallest index
    for ind in range(begin_ind,end_ind):
        if data[ind][2] >= target_high:       
            return 1
        elif data[ind][2] <= target_low:
            return -1

def search_stop(data,target_high,target_low,freq,ind,is_first=False): #freq means freq 60,30 etc, ind is the initial index for reference
    next_ind =  col_to_ind_dict[f'next_ind_{freq}']
    high_col = col_to_ind_dict[f'high_{freq}']
    low_col = col_to_ind_dict[f'low_{freq}']
    high_res = True if data[ind][high_col] >= target_high else False
    low_res = True if data[ind][low_col] <= target_low else False
 #   print('search_stop:',freq,data[ind][high_col],target_high,data[ind][low_col],target_low)
    if not high_res and not low_res: #BOTH NOT HIT
        if is_first:
            return 0
        else:
            return search_stop(data,target_high,target_low,freq,data[ind][next_ind])

    elif high_res and low_res: #BOTH HIT
        if freq != 1:
            freq = next_freq_ind_dict[freq]
            return search_stop(data,target_high,target_low,freq,ind)
        else:
            #since freq is 1 min,ind and  next_ind becomes the search range 
            return search_(data,ind,data[ind][next_ind],target_high,target_low)  # its search_()
    else:
        return 1 if high_res == True else -1 # 1 means target_high hit first








