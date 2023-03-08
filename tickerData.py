import yfinance as yf
import pandas as pd
from finta import TA

names = {'ta': TA}

def getTickerData(ticker, period, interval):
    hist = yf.download(tickers = ticker, period=period, interval=interval)
    df = pd.DataFrame(hist)
    df = df.reset_index()
    return df

def getTickerIndicatorData(ohlc_df, indicators):
    for indicator in indicators:
        ind_data = eval('TA.' + indicator + '(ohlc_df)',)
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        ohlc_df = ohlc_df.merge(ind_data, left_index=True, right_index=True)
        ohlc_df.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)
    return ohlc_df

def produce_prediction(df, window):  
    prediction = (df.shift(-window)['close'] >= df['close'])
    prediction = prediction.iloc[:-window]
    df['pred'] = prediction.astype(int)   
    return df