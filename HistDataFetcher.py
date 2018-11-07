import pickle
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf;yf.pdr_override()
import os


def get_data(ticker):
    data = pdr.get_data_yahoo(ticker, start='2016-01-01', end=earliest_recent)
    data = data[['Open', 'Volume']]
    data.columns = ['marketHigh', 'volume']
    pickle.dump(data, open('historicalStocksData/{}.p'.format(ticker), 'wb'))


tickers = [ticker.replace('.p', '') for ticker in os.listdir('latestStocksData')]
known_tickers = [ticker.replace('.p', '') for ticker in os.listdir('historicalStocksData')]
tickers = [ticker for ticker in tickers if ticker not in known_tickers]
earliest_recent = pickle.load(open('latestStocksData/AAPL.p', 'rb')).index[0]

for ticker in tickers:
    try:
        data = pdr.get_data_yahoo(ticker, start='2016-01-01', end=earliest_recent)
        data = data[['Open', 'Volume']]
        data.columns = ['marketHigh', 'volume']
        pickle.dump(data, open('historicalStocksData/{}.p'.format(ticker), 'wb'))
    except Exception:
        os.remove('latestStocksData/{}.p'.format(ticker))
