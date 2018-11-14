import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn;seaborn.set()
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import os
import time
import copy
from random import shuffle
import DataBase
from threading import Thread


class RealTime:
    def __init__(self):
        self.period = 5
        n_stocks = 100
        self.extremum_order = 5
        self.order_threshold = 0
        self.sell_trigger_long = 'zero crossing'  # 'zero crossing' or 'extremum'
        self.sell_trigger_short = 'zero crossing'
        self.cash = 5000000
        self.start_cash = 250000
        self.invested = 0

        offset = 9400
        self.past_data = pickle.load(open("FullData_5.p", "rb"))[:-offset]
        self.past_data = self.past_data[~self.past_data.index.duplicated(keep='last')]
        self.future_data = pickle.load(open("FullData_5.p", "rb"))[-offset:]

        self.tickers = list(self.past_data.columns)
        shuffle(self.tickers)
        self.tickers = self.tickers[:]
        self.past_data = self.past_data[self.tickers]
        self.future_data = self.future_data[self.tickers]

        self.positions = DataBase.get_positions(self.tickers)
        self.positions['Invested'][self.positions['Invested'] == 0] = self.start_cash
        self.positions['Provisioned'][self.positions['Provisioned'] == 0] = self.start_cash

    def get_indicators(self):
        for ticker in ['AAPL', 'MSFT', 'FB', 'AMZN']:
            data = pd.DataFrame(self.past_data[ticker]).dropna()
            if len(data) > 390 // self.period * 26:
                data['12'] = data.iloc[:, 0].ewm(span=12 * 390 // self.period).mean()
                data['26'] = data.iloc[:, 0].ewm(span=26 * 390 // self.period).mean()
                data['macd'] = data['12'] - data['26']
                data['signal'] = data['macd'].ewm(span=9 * 390 // self.period * 1.5).mean()
                data['diff'] = data['macd'] - data['signal']
                data['diff'] = StandardScaler(with_mean=False).fit_transform(data['diff'].values.reshape(-1, 1))

                subset = data['diff'].iloc[-self.extremum_order//2:]
                if self.positions.loc[ticker, 'LongPosition'] and self.sell_trigger_long == 'zero crossing' and data.iloc[-1, :]['diff'] > 0:
                    print('Sell', ticker)
                    new_money = data.iloc[-1, 0] / self.positions.loc[ticker, 'LongPosition'] * self.positions.loc[ticker, 'Invested']
                    self.update_position(ticker, long=0, invested=new_money)

                if self.positions.loc[ticker, 'ShortPosition'] and self.sell_trigger_short == 'zero crossing' and data.iloc[-1, :]['diff'] < 0:
                    print('Cover', ticker)
                    new_money = self.positions.loc[ticker, 'Provisioned'] / self.positions.loc[ticker, 'ShortPosition'] * (self.positions.loc[ticker, 'ShortPosition'] - data.iloc[-1, 0]) + self.positions.loc[ticker, 'Provisioned']
                    self.update_position(ticker, short=0, provisioned=new_money)

                if self.positions.loc[ticker, 'LongPosition'] or self.positions.loc[ticker, 'ShortPosition'] == 0:
                    positive_indices = subset.reset_index(drop=True).index[subset > 0].tolist()
                    local_maxima = argrelextrema(subset.values, np.greater, order=self.extremum_order)[0]
                    if len(list(set(local_maxima).intersection(positive_indices))):
                        # Reached a local positive maximum
                        if self.positions.loc[ticker, 'ShortPosition'] == 0 and data['diff'].iloc[-1] > self.order_threshold:
                            print('Short', ticker)
                            self.update_position(ticker, short=data.iloc[-1, 0])
                        if self.positions.loc[ticker, 'LongPosition']:
                            print('Sell', ticker)
                            new_money = data.iloc[-1, 0]/self.positions.loc[ticker, 'LongPosition'] * self.positions.loc[ticker, 'Invested']
                            self.update_position(ticker, long=0, invested=new_money)

                if self.positions.loc[ticker, 'LongPosition'] == 0 or self.positions.loc[ticker, 'ShortPosition']:
                    negative_indices = subset.reset_index(drop=True).index[subset < 0].tolist()
                    local_minima = argrelextrema(subset.values, np.less, order=self.extremum_order)[0]
                    if len(list(set(local_minima).intersection(negative_indices))):
                        # Reached a local positive minimum
                        if self.positions.loc[ticker, 'LongPosition'] == 0 and data['diff'].iloc[-1] < -self.order_threshold:
                            print('Buy', ticker)
                            self.update_position(ticker, long=data.iloc[-1, 0])
                        if self.positions.loc[ticker, 'ShortPosition']:
                            print('Cover', ticker)
                            new_money = self.positions.loc[ticker, 'Provisioned'] / self.positions.loc[ticker, 'ShortPosition'] * (self.positions.loc[ticker, 'ShortPosition'] - data.iloc[-1, 0]) + self.positions.loc[ticker, 'Provisioned']
                            self.update_position(ticker, short=0, provisioned=new_money)

        return data

    def ingest(self, data_line):
        self.past_data = self.past_data.append(data_line)

    def update_position(self, ticker, long=None, invested=None, short=None, provisioned=None):
        long = self.positions.loc[ticker, 'LongPosition'] if long is None else long
        invested = self.positions.loc[ticker, 'Invested'] if invested is None else invested
        short = self.positions.loc[ticker, 'ShortPosition'] if short is None else short
        provisioned = self.positions.loc[ticker, 'Provisioned'] if provisioned is None else provisioned

        print([long, invested, short, provisioned])
        self.positions.loc[ticker] = [long, invested, short, provisioned]
        print(self.positions.loc[ticker])
        Thread(target=DataBase.update_position, args=(ticker, long, invested, short, provisioned)).start()

rt = RealTime()
for i in range(100000):
    rt.ingest(rt.future_data.iloc[i])
    start = time.time()
    data = rt.get_indicators()
    print(time.time()-start)
    plt.clf()
    data['diff'].reset_index(drop=True).plot()
    plt.pause(0.0001)
plt.show()
data_path = ''

