import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn;seaborn.set()
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import os

pd.options.mode.chained_assignment = None


def simulate(data, fee, fee_flat, order_threshold, plot=False):
    data['12'] = data['average'].ewm(span=12*390).mean()
    data['26'] = data['average'].ewm(span=26*390).mean()
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
    data['long_buy_orders'][data['minima'] < -order_threshold] = 2

    data['short_buy_orders'] = pd.Series(np.nan)
    data['short_sell_orders'] = pd.Series(np.nan)
    data['short_sell_orders'][data['maxima'] > order_threshold] = 2

    for date, value in data['long_buy_orders'].iteritems():
        if data['long_buy_orders'][date] == 2:
            date_plus_1 = date + timedelta(minutes=1)
            abort_date = data.index[-1]
            sell_date = data['maxima'][date_plus_1:].first_valid_index()

            if data.index.get_loc(date) + 3*390 < len(data):
                date_plus_3 = date + timedelta(days=3)
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
            date_plus_1 = date + timedelta(minutes=1)
            abort_date = data.index[-1]
            sell_date = data['minima'][date_plus_1:].first_valid_index()

            if data.index.get_loc(date) + 3*390 < len(data):
                date_plus_3 = date + timedelta(days=3)
                if (data['diff'][date_plus_3:] > data['maxima'][date]).any():
                    abort_date = (data['diff'][date_plus_3:] > data['maxima'][date]).idxmax()

            if sell_date is not None:
                sell_date = min(abort_date, sell_date)
                data['short_buy_orders'][sell_date] = 2
                data['short_sell_orders'][list(data['maxima'][date_plus_1:sell_date].dropna().index)] = 1
            else:
                data['short_sell_orders'][date:] = np.nan
                
    shift = 1
    data['long_buy_orders'] = data['long_buy_orders'].shift(shift)
    data['short_sell_orders'] = data['short_sell_orders'].shift(shift)
    data['long_sell_orders'] = data['long_sell_orders'].shift(shift)
    data['short_buy_orders'] = data['short_buy_orders'].shift(shift)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        data = data.reset_index()
        data['average'].plot(ax=ax1)
        ax2.plot(data[['macd', 'signal']])

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
        plt.pause(0.0001)

    if data['long_buy_orders'].any():
        long_buy_prices = data['average'][data['long_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat
        long_sell_prices = data['average'][data['long_sell_orders'] == 2].reset_index(drop=True) * (1 - fee) - fee_flat

        print('ROI on long positions:', round(((long_sell_prices / long_buy_prices).cumprod().iloc[-1] - 1) * 100, 1), '%')

    if data['short_sell_orders'].any():
        short_buy_prices = data['average'][data['short_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat
        short_sell_prices = data['average'][data['short_sell_orders'] == 2].reset_index(drop=True) * (1 - fee) - fee_flat

        print('ROI on short positions:', round(((short_sell_prices / short_buy_prices).cumprod().iloc[-1] - 1) * 100, 1), '%')

    if data['short_sell_orders'].any() and data['long_sell_orders'].any():
        sell_prices = pd.Series(list(long_sell_prices) + list(short_sell_prices))
        buy_prices = pd.Series(list(long_buy_prices) + list(short_buy_prices))
    elif data['short_sell_orders'].any():
        sell_prices = pd.Series(short_sell_prices)
        buy_prices = pd.Series(short_buy_prices)
    elif data['long_buy_orders'].any():
        sell_prices = pd.Series(long_sell_prices)
        buy_prices = pd.Series(long_buy_prices)
    else:
        sell_prices = pd.Series()
        buy_prices = pd.Series()

    if len(sell_prices):
        n_years = len(data) / (250*390)
        yearly_factor = (sell_prices / buy_prices).cumprod().iloc[-1] ** (1 / n_years)
        annualized_return = round(100 * (yearly_factor - 1), 1)

        return annualized_return
    return 0


fee = 0.00
fee_flat = 0
order_threshold = 0.25

results = []
tickers = os.listdir('stocksData')[:]
# tickers = ['BLNK.p']
for stock in tickers:
    print(stock.split('.p')[0])
    print('{}/{}'.format(tickers.index(stock)+1, len(tickers)))
    data = pickle.load(open("stocksData/{}".format(stock), "rb"))
    if len(data) > 390*26:
        annualized_return = simulate(data, fee, fee_flat, order_threshold, plot=False)
        results.append((stock.split('.p')[0], annualized_return))

results = pd.DataFrame(results, columns=['Stock', 'Yearly interest rate equivalent']).set_index('Stock')
results = results.sort_values('Yearly interest rate equivalent', ascending=False)
results.to_csv('results2.csv')
print(results)
print('Mean rate : ', round(results['Yearly interest rate equivalent'].mean(), 1), '%')
# results.plot(kind='bar')
plt.show()