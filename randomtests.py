import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn;seaborn.set()
from scipy.signal import argrelextrema
from datetime import datetime, timedelta

data = pickle.load(open("aapl.p", "rb"))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

data['12'] = data['average'].ewm(span=12*390*1).mean()
data['26'] = data['average'].ewm(span=26*390*1).mean()
data['macd'] = data['12'] - data['26']
data['signal'] = data['macd'].ewm(span=9*390).mean()
data['diff'] = data['macd'] - data['signal']
data['diff'] = StandardScaler(with_mean=False).fit_transform(data['diff'].values.reshape(-1, 1))
signals = data['diff'].fillna(0).values

negative_indices = data['diff'].reset_index(drop=True).index[data['diff'] < 0].tolist()
local_minima = argrelextrema(data['diff'].values, np.less)[0]
indices = set(local_minima).intersection(negative_indices)
data['minima'] = pd.Series(np.nan)
data['minima'][indices] = data['diff'][indices]

positive_indices = data['diff'].reset_index(drop=True).index[data['diff'] > 0].tolist()
local_maxima = argrelextrema(data['diff'].values, np.greater)[0]
indices = set(local_maxima).intersection(positive_indices)
data['maxima'] = pd.Series(np.nan)
data['maxima'][indices] = data['diff'][indices]

data['long_buy_orders'] = pd.Series(np.nan)
data['long_sell_orders'] = pd.Series(np.nan)
data['long_buy_orders'][data['minima'] < -0.5] = 2

data['short_buy_orders'] = pd.Series(np.nan)
data['short_sell_orders'] = pd.Series(np.nan)
data['short_sell_orders'][data['maxima'] > 0.5] = 2

for date, value in data['long_buy_orders'].iteritems():
    if data['long_buy_orders'][date] == 2:
        date_plus_1 = date + timedelta(days=1)
        abort_date = data.index[-1]
        sell_date = data['maxima'][date_plus_1:].first_valid_index()

        if data.index.get_loc(date) + 3*390 < len(data):
            date_plus_3 = data.index[data.index.get_loc(date) + 3*390]
            if (data['diff'][date_plus_3:] < data['minima'][date]).any():
                abort_date = (data['diff'][date_plus_3:] < data['minima'][date]).idxmax()

        if sell_date is not None:
            sell_date = min(abort_date, sell_date)
            data['long_sell_orders'][sell_date] = 2
            data['long_buy_orders'][list(data['minima'][date_plus_1:sell_date].dropna().index)] = 1
        else:
            data['long_buy_orders'][date:] = np.nan

for date, value in data['short_sell_orders'].iteritems():
    if data['short_sell_orders'][date] == 2:
        date_plus_1 = date + timedelta(days=1)
        abort_date = data.index[-1]
        sell_date = data['minima'][date_plus_1:].first_valid_index()

        if data.index.get_loc(date) + 3*390*2 < len(data):
            date_plus_3 = data.index[data.index.get_loc(date) + 3*390*2]
            if (data['diff'][date_plus_3:] > data['maxima'][date]).any():
                abort_date = (data['diff'][date_plus_3:] > data['maxima'][date]).idxmax()

        if sell_date is not None:
            sell_date = min(abort_date, sell_date)
            data['short_buy_orders'][sell_date] = 2
            data['short_sell_orders'][list(data['maxima'][date_plus_1:sell_date].dropna().index)] = 1
        else:
            data['short_sell_orders'][date:] = np.nan

print(data)
data = data.reset_index()
data['average'].plot(ax=ax1)
ax2.plot(data[['macd', 'signal']])

signals = data['diff'].diff().fillna(0).values
ax2.fill_between(list(data.index), signals)

for date, value in data['long_buy_orders'].iteritems():
    if value == 2:
        ax1.axvline(x=date, color='blue', linewidth=0.5)
        ax2.axvline(x=date, color='blue', linewidth=0.5)
for date, value in data['long_sell_orders'].iteritems():
    if value == 2:
        ax1.axvline(x=date, color='cyan', linewidth=0.5)
        ax2.axvline(x=date, color='cyan', linewidth=0.5)

for date, value in data['short_buy_orders'].iteritems():
    if value == 2:
        ax1.axvline(x=date, color='orange', linewidth=0.5)
        ax2.axvline(x=date, color='orange', linewidth=0.5)
for date, value in data['short_sell_orders'].iteritems():
    if value == 2:
        ax1.axvline(x=date, color='red', linewidth=0.5)
        ax2.axvline(x=date, color='red', linewidth=0.5)

plt.show()