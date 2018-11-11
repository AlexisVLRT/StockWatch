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


def simulate(data, period, fee, fee_flat, start_cash, order_threshold, verif_size, order_shift, extremums_order, min_days_before_abort, sell_trigger_long, sell_trigger_short, allow_long=True, allow_short=True, plot=False):
    data = data.drop_duplicates()
    data['12'] = data['open'].ewm(span=12*390//period).mean()
    data['26'] = data['open'].ewm(span=26*390//period).mean()
    data['macd'] = data['12'] - data['26']
    data['signal'] = data['macd'].ewm(span=9*390//period*1.5).mean()
    data['diff'] = data['macd'] - data['signal']

    data['diff'] = StandardScaler(with_mean=False).fit_transform(data['diff'].values.reshape(-1, 1))
    signals = data['diff'].fillna(0).values

    negative_indices = data['diff'].reset_index(drop=True).index[data['diff'] < 0].tolist()
    local_minima = argrelextrema(data['diff'].values, np.less, order=extremums_order)[0]
    indices = list(set(local_minima).intersection(negative_indices))
    data['minima'] = pd.Series(np.nan)
    data['minima'].iloc[indices] = data['diff'].iloc[indices]

    positive_indices = data['diff'].reset_index(drop=True).index[data['diff'] > 0].tolist()
    local_maxima = argrelextrema(data['diff'].values, np.greater, order=extremums_order)[0]
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

    data['immobilized_long'] = 0
    data['immobilized_short'] = 0

    for date, value in data['long_buy_orders'].iteritems():
        if data['long_buy_orders'].loc[date] == 2:
            date_plus_1 = int(datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(minutes=period), '%Y%m%d%H%M'))
            abort_date = data.index[-1]

            if sell_trigger_long == 'extremum':
                sell_date = data['maxima'].loc[date_plus_1:].first_valid_index()
            elif sell_trigger_long == 'zero crossing':
                sell_date = data['>zero'].loc[date_plus_1:].first_valid_index()

            if data.index.get_loc(date) + 3*390//period < len(data):
                date_plus_3 = int(datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(days=min_days_before_abort), '%Y%m%d%H%M'))
                if (data['diff'].loc[date_plus_3:] < data['minima'].loc[date]).any():
                    abort_date = (data['diff'].loc[date_plus_3:] < data['minima'].loc[date]).idxmax()

            if sell_date is not None:
                sell_date = min(abort_date, sell_date)
                data.loc[sell_date, 'long_sell_orders'] = 2
                data['long_buy_orders'][list(data['minima'].loc[date_plus_1:sell_date].dropna().index)] = 1
                data.loc[date:sell_date, 'immobilized_long'] += 1
            else:
                data.loc[date:, 'long_buy_orders'] = np.nan

    for date, value in data['short_sell_orders'].iteritems():
        if data['short_sell_orders'].loc[date] == 2:
            date_plus_1 = int(datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(minutes=period), '%Y%m%d%H%M'))
            abort_date = data.index[-1]

            if sell_trigger_short == 'extremum':
                sell_date = data['minima'].loc[date_plus_1:].first_valid_index()
            elif sell_trigger_short == 'zero crossing':
                sell_date = data['<zero'].loc[date_plus_1:].first_valid_index()

            if data.index.get_loc(date) + 3*390//period < len(data):
                date_plus_3 = int(datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(days=min_days_before_abort), '%Y%m%d%H%M'))
                if (data['diff'].loc[date_plus_3:] > data['maxima'].loc[date]).any():
                    abort_date = (data['diff'].loc[date_plus_3:] > data['maxima'].loc[date]).idxmax()

            if sell_date is not None:
                sell_date = min(abort_date, sell_date)
                data.loc[sell_date, 'short_buy_orders'] = 2
                data['short_sell_orders'][list(data['maxima'].loc[date_plus_1:sell_date].dropna().index)] = 1
                data.loc[date:sell_date, 'immobilized_short'] += 1
            else:
                data.loc[date:, 'short_sell_orders'] = np.nan
                
    shift = order_shift
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
        # plt.pause(0.00001)
        plt.show()

    short_profit_first, short_profit_second, long_profit_first, long_profit_second = 0, 0, 0, 0
    if data['long_buy_orders'].any():
        long_buy_prices_first = data['open'].iloc[:int(verif_size * len(data))][data['long_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat
        long_buy_prices_second = data['open'].iloc[int(verif_size * len(data)):][data['long_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat
        # print(long_buy_prices_first, long_buy_prices_second)

        long_sell_prices_first = (data['open'][data['long_sell_orders'] == 2] * (1 - fee) - fee_flat)[:len(long_buy_prices_first)].reset_index(drop=True)
        long_sell_prices_second = (data['open'][data['long_sell_orders'] == 2] * (1 - fee) - fee_flat)[len(long_buy_prices_first):].reset_index(drop=True)

        if len(long_sell_prices_first):
            cumulated = (long_sell_prices_first / long_buy_prices_first[:len(long_sell_prices_first)]).cumprod()
            immobilized_clusters = data['immobilized_long'].groupby([(data['immobilized_long'] != data['immobilized_long'].shift())]).get_group(True)
            immobilized_clusters = immobilized_clusters.drop(immobilized_clusters.index[0]) if immobilized_clusters.iloc[0] == 0 else immobilized_clusters

            for i in range(0, len(immobilized_clusters), 2):
                if i < len(cumulated):
                    data.loc[immobilized_clusters.index[i]:immobilized_clusters.index[i+1], 'immobilized_long'] += start_cash * cumulated.iloc[i//2]
            long_profit_first = int((cumulated.iloc[-1]) * start_cash - start_cash)

        if len(long_sell_prices_second):
            cumulated = (long_sell_prices_second / long_buy_prices_second[:len(long_sell_prices_second)]).cumprod()
            immobilized_clusters = data['immobilized_long'].groupby([(data['immobilized_long'] != data['immobilized_long'].shift())]).get_group(True)
            immobilized_clusters = immobilized_clusters.drop(immobilized_clusters.index[0]) if immobilized_clusters.iloc[0] == 0 else immobilized_clusters

            for i in range(0, len(immobilized_clusters), 2):
                if i < len(cumulated):
                    data.loc[immobilized_clusters.index[i]:immobilized_clusters.index[i + 1], 'immobilized_long'] += start_cash * cumulated.iloc[i // 2]
            long_profit_second = int((cumulated.iloc[-1]) * start_cash - start_cash)

        print('Profit on long positions:', long_profit_first, long_profit_second)

    if data['short_sell_orders'].any():
        short_buy_prices_first = data['open'].iloc[:int(verif_size * len(data))][data['short_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat
        short_buy_prices_second = data['open'].iloc[int(verif_size * len(data)):][data['short_buy_orders'] == 2].reset_index(drop=True) * (1 + fee) + fee_flat

        short_sell_prices_first = (data['open'][data['short_sell_orders'] == 2] * (1 - fee) - fee_flat)[:len(short_buy_prices_first)].reset_index(drop=True)
        short_sell_prices_second = (data['open'][data['short_sell_orders'] == 2] * (1 - fee) - fee_flat)[len(short_buy_prices_first):].reset_index(drop=True)

        recap_first = pd.DataFrame([short_sell_prices_first[:len(short_buy_prices_first)], short_buy_prices_first]).reset_index(drop=True).T
        recap_first.columns = ['Sell', 'Buy']
        recap_first['diff'] = recap_first['Sell'] - recap_first['Buy']
        recap_first['money'] = pd.Series()
        for i in range(len(recap_first)):
            if i == 0:
                recap_first.loc[recap_first.index[i], 'money'] = start_cash + start_cash % recap_first.iloc[i]['Sell'] + (start_cash // recap_first.iloc[i]['Sell']) * recap_first.iloc[i]['diff']
            else:
                recap_first.loc[recap_first.index[i], 'money'] = recap_first.iloc[i - 1]['money'] + recap_first.iloc[i - 1]['money'] % recap_first.iloc[i]['Sell'] + (recap_first.iloc[i - 1]['money'] // recap_first.iloc[i]['Sell']) * recap_first.iloc[i]['diff']

        lost_more_than_invested = recap_first['money'] + start_cash < 0
        if lost_more_than_invested.sum():
            lost_more_than_invested = lost_more_than_invested.index[0]
            recap_first = recap_first[:lost_more_than_invested+1]
        if len(recap_first):
            immobilized_clusters = data['immobilized_short'].groupby([(data['immobilized_short'] != data['immobilized_short'].shift())]).get_group(True)
            immobilized_clusters = immobilized_clusters.drop(immobilized_clusters.index[0]) if immobilized_clusters.iloc[0] == 0 else immobilized_clusters

            for i in range(0, len(immobilized_clusters), 2):
                data.loc[immobilized_clusters.index[i]:immobilized_clusters.index[i+1], 'immobilized_short'] += recap_first['money'].iloc[i // 2 - 1]
            data.loc[immobilized_clusters.index[0]:immobilized_clusters.index[1], 'immobilized_short'] = start_cash
            short_profit_first = int(recap_first.loc[recap_first.index[-1], 'money'] - start_cash)

        recap_second = pd.DataFrame([short_sell_prices_second[:len(short_buy_prices_second)], short_buy_prices_second]).reset_index(drop=True).T
        recap_second.columns = ['Sell', 'Buy']
        recap_second['diff'] = recap_second['Sell'] - recap_second['Buy']
        recap_second['money'] = pd.Series()
        for i in range(len(recap_second)):
            if i == 0:
                recap_second.loc[recap_second.index[i], 'money'] = start_cash + start_cash % recap_second.iloc[i]['Sell'] + (start_cash // recap_second.iloc[i]['Sell']) * recap_second.iloc[i]['diff']
            else:
                recap_second.loc[recap_second.index[i], 'money'] = recap_second.iloc[i - 1]['money'] + recap_second.iloc[i - 1]['money'] % recap_second.iloc[i]['Sell'] + (recap_second.iloc[i - 1]['money'] // recap_second.iloc[i]['Sell']) * recap_second.iloc[i]['diff']

        if len(recap_second):
            immobilized_clusters = data['immobilized_short'].groupby([(data['immobilized_short'] != data['immobilized_short'].shift())]).get_group(True)
            immobilized_clusters = immobilized_clusters.drop(immobilized_clusters.index[0]) if immobilized_clusters.iloc[0] == 0 else immobilized_clusters

            for i in range(0, len(immobilized_clusters), 2):
                data.loc[immobilized_clusters.index[i]:immobilized_clusters.index[i + 1], 'immobilized_short'] += recap_second['money'].iloc[i // 2 - 1]
            data.loc[immobilized_clusters.index[0]:immobilized_clusters.index[1], 'immobilized_short'] = start_cash
            short_profit_second = int(recap_second.loc[recap_second.index[-1], 'money'] - start_cash)

        # print('ROI on short positions:', round(((short_sell_prices_first / short_buy_prices_first).cumprod().iloc[-1] - 1) * 100, 1), '%')
        print('Profit on short positions:', short_profit_first, short_profit_second)

    yearly_profit_first = round((short_profit_first + long_profit_first), 1)
    yearly_profit_second = round((short_profit_second + long_profit_second), 1)
    print('TOTAL : ', yearly_profit_first + yearly_profit_second)
    return yearly_profit_first, yearly_profit_second, data['immobilized_short'].add(data['immobilized_long'])


# Change this to the path where your pickled stocks data is
data_path = 'D:/PickledStocksData'

# Simulation parameters
period = 5
fee = 10/25000
fee_flat = 0
start_cash = 50000
order_threshold = 1.25
verif_size = 1
order_shift = 15
extremums_order = 5
min_days_before_abort = 5
sell_trigger_long = 'zero crossing'  # 'zero crossing' or 'extremum'
sell_trigger_short = 'zero crossing'
plot = False

results = []
immobilized_funds = None
tickers = [file for file in os.listdir(data_path) if '_'+str(period)+'.' in file][:]
# tickers = ['GEVO_5.p', 'AAPL_5.p', 'FB_5.p', 'MSFT_5.p']
for stock in tickers:
    print(stock.split('.p')[0])
    print('{}/{}'.format(tickers.index(stock)+1, len(tickers)))
    data = pickle.load(open(data_path + "/{}".format(stock), "rb"))
    if len(data) > 390//period*26 and stock not in []:
        yearly_profit_first, yearly_profit_second, immobilized = simulate(data, period, fee, fee_flat, start_cash, order_threshold, verif_size, order_shift, extremums_order, min_days_before_abort, sell_trigger_long, plot=plot)
        immobilized_funds = immobilized_funds.add(immobilized, fill_value=0) if immobilized_funds is not None else immobilized
        results.append((stock.split('.p')[0], yearly_profit_first, yearly_profit_second))

results = pd.DataFrame(results, columns=['Stock', 'yearly_profit_first', 'yearly_profit_second']).set_index('Stock')
results = results.sort_values('yearly_profit_first', ascending=False)
results.to_csv('results2.csv')
print(results)
print(results.describe())

total_profit = round(results['yearly_profit_first'].add(results['yearly_profit_first']).sum(), 1)
max_immo = int(max(immobilized_funds))
print('Total profit :', total_profit)
print('Max immobilized funds :', max_immo)
print('Interest rate over period :', round(100*total_profit/max_immo, 1), '%')
immobilized_funds.reset_index(drop=True).plot()
plt.show()