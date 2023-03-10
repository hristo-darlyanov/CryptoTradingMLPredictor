import yfinance as yf
import pandas as pd
from finta import TA

names = {'ta': TA}

def getTickerData(ticker, period, interval):
    hist = yf.download(tickers = ticker, period=period, interval=interval)
    df = pd.DataFrame(hist)
    df = df.reset_index()
    return df

def getTickerIndicatorData(ohlc_df, indicators, dropna = False):
    for indicator in indicators:
        ind_data = eval('TA.' + indicator + '(ohlc_df)',)
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        ohlc_df = ohlc_df.merge(ind_data, left_index=True, right_index=True)
        ohlc_df.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)
    if dropna == False:   
        return ohlc_df
    else:
        return ohlc_df.dropna()

def produce_prediction(df, window, dropna = False, removeohl = False):  
    prediction = (df.shift(-window)['close'] >= df['close'])
    prediction = prediction.iloc[:-window]
    df['pred'] = prediction.astype(int)

    if removeohl == True:
        df.drop(columns={'open', 'high', 'low'}, inplace=True)

    if dropna == False:   
        return df
    else:
        return df.dropna()