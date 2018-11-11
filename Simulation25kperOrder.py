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

pd.options.mode.chained_assignment = None


def simulate(data, period, fee, fee_flat, order_threshold, verif_size=0.8, allow_long=True, allow_short=True, plot=False):
    data = data.drop_duplicates()
    data['12'] = data['open'].ewm(span=12*390//period).mean()
    data['26'] = data['open'].ewm(span=26*390//period).mean()
    data['macd'] = data['12'] - data['26']
    data['signal'] = data['macd'].ewm(span=9*390//period*1.5).mean()
    data['diff'] = data['macd'] - data['signal']

    data['diff'] = StandardScaler(with_mean=False).fit_transform(data['diff'].values.reshape(-1, 1))
    signals = data['diff'].fillna(0).values

    negative_indices = data['diff'].reset_index(drop=True).index[data['diff'] < 0].tolist()
    local_minima = argrelextrema(data['diff'].values, np.less, order=5)[0]
    indices = list(set(local_minima).intersection(negative_indices))
    data['minima'] = pd.Series(np.nan)
    data['minima'].iloc[indices] = data['diff'].iloc[indices]

    positive_indices = data['diff'].reset_index(drop=True).index[data['diff'] > 0].tolist()
    local_maxima = argrelextrema(data['diff'].values, np.greater, order=5)[0]
    indices = list(set(local_maxima).intersection(positive_indices))
    data['maxima'] = pd.Series(np.nan)
    data['maxima'].iloc[indices] = data['diff'].iloc[indices]

    data['>zero'] = pd.Series(np.nan)
    data['>zero'] = data['diff'][data['diff'] > 0]
    data['<zero'] = pd.Series(np.nan)
    data['<zero'] = data['diff'][data['diff'] < 0]


    data['long_buy_orders'] = pd.Series(np.nan)
    data['long_sell_orders'] = pd.Series(np.nan)
    if allow_long:
        data['long_buy_orders'][data['minima'] < -order_threshold] = 2

    data['short_buy_orders'] = pd.Series(np.nan)
    data['short_sell_orders'] = pd.Series(np.nan)
    if allow_short:
        data['short_sell_orders'][data['maxima'] > order_threshold] = 2

    data['immobilized'] = 0

    for date, value in data['long_buy_orders'].iteritems():
        if data['long_buy_orders'].loc[date] == 2:
            date_plus_1 = int(datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(minutes=period), '%Y%m%d%H%M'))
            abort_date = data.index[-1]

            # sell_date = data['maxima'].loc[date_plus_1:].first_valid_index()
            sell_date = data['>zero'].loc[date_plus_1:].first_valid_index()

            if data.index.get_loc(date) + 3*390//period < len(data):
                date_plus_3 = int(datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(days=3), '%Y%m%d%H%M'))
                if (data['diff'].loc[date_plus_3:] < data['minima'].loc[date]).any():
                    abort_date = (data['diff'].loc[date_plus_3:] < data['minima'].loc[date]).idxmax()

            if sell_date is not None:
                sell_date = min(abort_date, sell_date)
                data['long_sell_orders'].loc[sell_date] = 2
                data['long_buy_orders'][list(data['minima'].loc[date_plus_1:sell_date].dropna().index)] = 1
                data['immobilized'].loc[date:sell_date] += 1
            else:
                data['long_buy_orders'].loc[date:] = np.nan

    for date, value in data['short_sell_orders'].iteritems():
        if data['short_sell_orders'].loc[date] == 2:
            date_plus_1 = int(
                datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(minutes=period), '%Y%m%d%H%M'))
            abort_date = data.index[-1]
            # sell_date = data['minima'].loc[date_plus_1:].first_valid_index()
            sell_date = data['<zero'].loc[date_plus_1:].first_valid_index()

            if data.index.get_loc(date) + 3*390//period < len(data):
                date_plus_3 = int(
                    datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(days=3), '%Y%m%d%H%M'))
                if (data['diff'].loc[date_plus_3:] > data['maxima'].loc[date]).any():
                    abort_date = (data['diff'].loc[date_plus_3:] > data['maxima'].loc[date]).idxmax()

            if sell_date is not None:
                sell_date = min(abort_date, sell_date)
                data['short_buy_orders'].loc[sell_date] = 2
                data['short_sell_orders'][list(data['maxima'].loc[date_plus_1:sell_date].dropna().index)] = 1
                data['immobilized'].loc[date:sell_date] += 1
            else:
                data['short_sell_orders'].loc[date:] = np.nan
                
    shift = 15
    data['long_buy_orders'] = data['long_buy_orders'].shift(shift)
    data['short_sell_orders'] = data['short_sell_orders'].shift(shift)
    data['long_sell_orders'] = data['long_sell_orders'].shift(shift)
    data['short_buy_orders'] = data['short_buy_orders'].shift(shift)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        data = data.reset_index()
        data['open'].plot(ax=ax1)
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

    short_profit_first, short_profit_second, long_profit_first, long_profit_second = 0, 0, 0, 0
    if data['long_buy_orders'].any():
        long_buy_prices_first = data['open'].iloc[:int(verif_size * len(data))][data['long_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat
        long_buy_prices_second = data['open'].iloc[int(verif_size * len(data)):][data['long_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat
        # print(long_buy_prices_first, long_buy_prices_second)

        long_sell_prices_first = (data['open'][data['long_sell_orders'] == 2] * (1 - fee) - fee_flat)[:len(long_buy_prices_first)].reset_index(drop=True)
        long_sell_prices_second = (data['open'][data['long_sell_orders'] == 2] * (1 - fee) - fee_flat)[len(long_buy_prices_first):].reset_index(drop=True)
        # print(long_sell_prices_first, long_sell_prices_second)

        if len(long_sell_prices_first):
            long_profit_first = round((((long_sell_prices_first - long_buy_prices_first)*(25000//long_buy_prices_first)).cumsum().iloc[-1]), 1)
        if len(long_sell_prices_second):
            long_profit_second = round((((long_sell_prices_second - long_buy_prices_second) * (25000 // long_buy_prices_second)).cumsum().iloc[-1]), 1)

        # print('ROI on long positions:', round(((long_sell_prices_first / long_buy_prices_first).cumprod().iloc[-1] - 1) * 100, 1), '%')
        print('Profit on long positions:', long_profit_first, long_profit_second)

    if data['short_sell_orders'].any():
        short_buy_prices_first = data['open'].iloc[:int(verif_size * len(data))][data['short_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat
        short_buy_prices_second = data['open'].iloc[int(verif_size * len(data)):][data['short_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat

        short_sell_prices_first = (data['open'][data['short_sell_orders'] == 2] * (1 - fee) - fee_flat)[:len(short_buy_prices_first)].reset_index(drop=True)
        short_sell_prices_second = (data['open'][data['short_sell_orders'] == 2] * (1 - fee) - fee_flat)[len(short_buy_prices_first):].reset_index(drop=True)

        if len(short_sell_prices_first):
            short_profit_first = round((((short_sell_prices_first - short_buy_prices_first) * (25000 // short_sell_prices_first)).cumsum().iloc[-1]), 1)
        if len(short_sell_prices_second):
            short_profit_second = round((((short_sell_prices_second - short_buy_prices_second) * (25000 // short_sell_prices_second)).cumsum().iloc[-1]), 1)

        # print('ROI on short positions:', round(((short_sell_prices_first / short_buy_prices_first).cumprod().iloc[-1] - 1) * 100, 1), '%')
        print('Profit on short positions:', short_profit_first, short_profit_second)

    # if data['short_sell_orders'].any() and data['long_sell_orders'].any():
    #     sell_prices = pd.Series(list(long_sell_prices) + list(short_sell_prices))
    #     buy_prices = pd.Series(list(long_buy_prices) + list(short_buy_prices))
    # elif data['short_sell_orders'].any():
    #     sell_prices = pd.Series(short_sell_prices)
    #     buy_prices = pd.Series(short_buy_prices)
    # elif data['long_buy_orders'].any():
    #     sell_prices = pd.Series(long_sell_prices)
    #     buy_prices = pd.Series(long_buy_prices)
    # else:
    #     sell_prices = pd.Series()
    #     buy_prices = pd.Series()

    # if len(sell_prices):
    #     n_years = len(data) / (250*390//period)
    #     yearly_factor = (sell_prices / buy_prices).cumprod().iloc[-1] ** (1 / n_years)
    #
    #     yearly_profit = round((short_profit + long_profit) * n_years, 1)
    #     annualized_return = round(100 * (yearly_factor - 1), 1)
    #
    #     return annualized_return, yearly_profit, data['immobilized']
    # return 0, 0, data['immobilized']
    n_years = len(data) / (250 * 390 // period)
    yearly_profit_first = round((short_profit_first + long_profit_first), 1)
    yearly_profit_second = round((short_profit_second + long_profit_second), 1)
    return yearly_profit_first, yearly_profit_second, data['immobilized']


period = 5
fee = 10/25000
fee_flat = 0
order_threshold = 1.25

results = []
immobilized_funds = None
tickers = [file for file in os.listdir('D:/PickledStocksData') if '_'+str(period)+'.' in file][:]
# tickers = ['HMNY_5.p', 'ADBE_5.p']
for stock in tickers:
    print(stock.split('.p')[0])
    print('{}/{}'.format(tickers.index(stock)+1, len(tickers)))
    data = pickle.load(open("D:/PickledStocksData/{}".format(stock), "rb"))
    if len(data) > 390//period*26 and stock not in []:
        yearly_profit_first, yearly_profit_second, immobilized = simulate(data, period, fee, fee_flat, order_threshold, allow_short=True, plot=False)
        immobilized_funds = immobilized_funds.add(immobilized, fill_value=0) if immobilized_funds is not None else immobilized
        results.append((stock.split('.p')[0], yearly_profit_first, yearly_profit_second))

results = pd.DataFrame(results, columns=['Stock', 'yearly_profit_first', 'yearly_profit_second']).set_index('Stock')
results = results.sort_values('yearly_profit_first', ascending=False)
results.to_csv('results1.csv')
print(results)
print(results.describe())
# print('Mean rate : ', round(results['Yearly interest rate equivalent'].mean(), 1), '%')
print('Total profit : ', round(results['yearly_profit_first'].sum(), 1))
print(max(immobilized_funds)*25000)
print(results['yearly_profit_first'].sum()*100 / (max(immobilized_funds)*25000))
immobilized_funds.reset_index(drop=True).plot()
plt.show()