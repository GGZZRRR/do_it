# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 21:35:46 2021

@author: user
"""




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from glob import glob
import os
import copy as cp
import time

import statsmodels.api as sm
def ff_regrr_beta(x, xname, yname):

    ax = x[xname]
    ay = x[yname]
    
    model = sm.OLS(ay, ax, missing='drop', hasconst=False)   
    results = model.fit()
    return  results.params[0]

def cal_corr(df_data, xname, yname):
    dfx = df_data.groupby("time_id")[[xname,yname]].corr()
    dfx .reset_index(inplace=True)
    dfx = dfx.groupby("time_id").head(1)
    dfx["corr"] = dfx[yname]    
    return dfx[["time_id","corr"]]

def calc_rollingstats(rolling_x, roll_name):
    #统计量
    if len(rolling_x) > 0 :
        roll_autocorr =  rolling_x.groupby("time_id")[[roll_name,"xpre"]].corr()
        roll_autocorr .reset_index(inplace=True)
        roll_autocorr = roll_autocorr.groupby("time_id").head(1)
        roll_autocorr.index = roll_autocorr["time_id"]
        del roll_autocorr["time_id"]
        
        roll_autocorr = pd.DataFrame({roll_name+"_autocorr": roll_autocorr["xpre"]})
        
        roll_mean = pd.DataFrame({roll_name+"_mean": rolling_x.groupby("time_id")[roll_name].mean()})
        roll_std = pd.DataFrame({roll_name+"_std": rolling_x.groupby("time_id")[roll_name].std()})
        roll_skew = pd.DataFrame({roll_name+"_skew": rolling_x.groupby("time_id")[roll_name].skew()})
        
        data_merge = pd.merge(roll_mean, roll_std, left_index=True, right_index=True, how = "inner")
        data_merge = pd.merge(data_merge , roll_skew, left_index=True, right_index=True, how = "inner")
        data_merge = pd.merge(data_merge , roll_autocorr, left_index=True, right_index=True, how = "inner")

    else:
    
        data_merge = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]])
        data_merge.columns = [roll_name + "_mean", roll_name + "_std", roll_name + "_skew", roll_name + "_autocorr"]
    
    return data_merge

        
def make_Kline(df_data, price_name, vol_name, amt_name, mini_tick):
    
    df_data["gi"] = df_data["seconds_in_bucket"]/mini_tick
    df_data["gi"] = df_data["gi"] .astype(int)
    
    df_data["pre"] = df_data.groupby(["time_id","gi"])[price_name].shift(1)
    df_data["ret"] = df_data[price_name] / df_data["pre"] - 1
    df_data["absret"] = abs(df_data["ret"] )
    df_retsum = pd.DataFrame({"retsum":df_data.groupby(["time_id","gi"])["ret"].sum()})
    df_absretsum = pd.DataFrame({"absretsum":df_data.groupby(["time_id","gi"])["absret"].sum()})
    
    
    df_amt =  pd.DataFrame({amt_name + "sum": df_data.groupby(["time_id","gi"])[amt_name].sum()})
    df_vol =  pd.DataFrame({vol_name + "sum": df_data.groupby(["time_id","gi"])[vol_name].sum()})
    
    df_mean = pd.DataFrame({price_name + "mean": df_data.groupby(["time_id","gi"])[price_name].mean()})
    df_high = pd.DataFrame({price_name + "high": df_data.groupby(["time_id","gi"])[price_name].max()})
    df_low = pd.DataFrame({price_name + "low": df_data.groupby(["time_id","gi"])[price_name].min()})
  
    
    df_candle = pd.merge(df_high, df_low, left_index=True, right_index=True,how="inner")
    df_candle = pd.merge(df_candle, df_mean, left_index=True, right_index=True,how="inner")

    df_candle = pd.merge(df_candle, df_vol, left_index=True, right_index=True,how="inner")
    df_candle = pd.merge(df_candle, df_amt, left_index=True, right_index=True,how="inner")    
    df_candle = pd.merge(df_candle, df_retsum, left_index=True, right_index=True,how="inner")    
    df_candle = pd.merge(df_candle, df_absretsum, left_index=True, right_index=True,how="inner")    
      
    
    df_candle.reset_index(inplace=True)
        
    df_open = df_data.groupby(["time_id","gi"]).head(1)
    df_open[price_name + "open"] = df_open[price_name]
    
    df_close = df_data.groupby(["time_id","gi"]).tail(1)
    df_close[price_name + "close"] = df_close[price_name]
    
    df_candle = pd.merge(df_candle, df_open[["time_id","gi",price_name + "open"]], on=["time_id","gi"], how="inner")
    df_candle = pd.merge(df_candle, df_close[["time_id","gi",price_name + "close"]], on=["time_id","gi"], how="inner")
           
    return df_candle
        
        
def cal_candlefactor(df_candle, price_name, vol_name, amt_name):
    f_name = price_name + "candle"
    #f1:illiq
    df_candle[f_name + "f1"] = (2 * (df_candle[price_name + "high"] - df_candle[price_name + "low"]) 
                    - abs(df_candle[price_name + "open"] - df_candle[price_name + "close"]))/df_candle[amt_name + "sum"]
    
    #f2 strength
    df_candle[f_name + "f2"] = df_candle["retsum"]/df_candle["absretsum"]
    #f3:ad
    df_candle[f_name + "f3"] =  (2 *df_candle[price_name + "close"] - df_candle[price_name + "low"]\
                    - df_candle[price_name + "high"] )/(df_candle[price_name + "high"] - df_candle[price_name + "low"]) \
                    * df_candle[vol_name + "sum"]
                    

    #f3: obv
    df_candle[price_name+"preclose"] = df_candle.groupby(["time_id"])[price_name+"close"].shift(1)
    df_candle["retx"] = df_candle[price_name+"close"]/df_candle[price_name+"preclose"]-1
    df_candle["obv"] = np.where(df_candle["retx"]>0, df_candle[vol_name + "sum"],
                         np.where(df_candle["retx"]<0, -df_candle[vol_name + "sum"],0))
       
    #atr
    df_candle["hl"] = df_candle[price_name+"high"] - df_candle[price_name+"low"]  
    df_candle["prec_h"] = abs(df_candle[price_name+"high"] - df_candle[price_name+"preclose"])
    df_candle["prec_l"] = abs(df_candle[price_name+"low"] - df_candle[price_name+"preclose"])
    
    df_candle["x"] = np.where(df_candle["hl"]>df_candle["prec_h"],df_candle["hl"],df_candle["prec_h"])  
    df_candle["art1"] = np.where(df_candle["x"]>df_candle["prec_l"],df_candle["x"],df_candle["prec_l"])
    
    df_factor = pd.DataFrame({"illiq": df_candle.groupby("time_id")[f_name + "f1"].sum(),
                          "strength": df_candle.groupby("time_id")[f_name + "f2"].sum(),
                          "ad": df_candle.groupby("time_id")[f_name + "f3"].sum(),
                          "obv": df_candle.groupby("time_id")["obv"].sum(),
                          "art1":df_candle.groupby("time_id")["art1"].mean(),})
                          
    return df_factor
        
        
        

def calculate_features1(book_df):
        
    book_df['wap'] = (book_df['bid_price1'] * book_df['ask_size1'] + book_df['ask_price1'] * 
                           book_df['bid_size1']) / (book_df['bid_size1']+ book_df['ask_size1'])
    book_df["vol_ab"] = book_df['bid_size1']+ book_df['ask_size1']
    book_df["amt_ab"] = book_df['bid_price1'] * book_df['ask_size1'] + book_df['ask_price1'] * book_df['bid_size1']
    
    book_df["amt_a"] = book_df['ask_price1'] * book_df['ask_size1'] 
    book_df["amt_b"] = book_df['bid_price1'] * book_df['bid_size1'] 
    


    for roll_window in [5, 10, 30, 60]:
        #rolling指标
        price_name = "wap"
        for price_name in ["wap","bid_price1","ask_price1"]:
            roll_name0 = price_name + "roll_std" 
            roll_name = roll_name0 + str(roll_window)
            
            rolling_x = pd.DataFrame({roll_name:book_df.groupby("time_id")[price_name].rolling(roll_window).std()})
            rolling_x.reset_index(inplace=True)
            rolling_x.loc[:,"xpre"] = rolling_x.groupby("time_id")[roll_name].shift(1)
            #计算统计量因子： 标准差的统计量
            df_factor = calc_rollingstats(rolling_x, roll_name)
            
            
    #ts_corr
    xname = "bid_price1"
    yname = "bid_size1"
    
    df_data = cp.deepcopy(book_df)    
    df_factor = cal_corr(df_data, xname, yname)
    
    #olseta    
    xname = "bid_price1"
    yname = "bid_size1"
    
    df_data = cp.deepcopy(book_df)
    df_factor = pd.DataFrame({"olsbeta":df_data.groupby("time_id").apply(lambda x:ff_regrr_beta(x, xname, yname) )})
    
    mini_tick = 10
    
    for price_name, vol_name, amt_name in [["wap","vol_ab","amt_ab"],
                                       ["bid_price1","bid_size1","amt_b"],
                                       ["ask_price1","ask_size1","amt_a"]]:
        df_data = cp.deepcopy(book_df)
        
        df_candle = make_Kline(df_data, price_name, vol_name, amt_name, mini_tick)
        
        df_factor = cal_candlefactor(df_candle, price_name, vol_name, amt_name)
        
        
        




