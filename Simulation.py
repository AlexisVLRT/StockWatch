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

pd.options.mode.chained_assignment = None


def simulate(data, period, fee, start_cash, remaining_funds, invested_funds, orders, order_threshold, verif_size, order_shift, extremums_order, min_days_before_abort, sell_trigger_long, sell_trigger_short, allow_long=True, allow_short=True, plot=False):
    data = data[~data.index.duplicated(keep='last')]
    # remaining_funds = remaining_funds.add(data['open']*0).fillna(method='ffill')
    # remaining_funds = remaining_funds[~remaining_funds.index.duplicated(keep='last')]
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

    data['long_buy_orders'] = data['long_buy_orders'].shift(order_shift)
    data['short_sell_orders'] = data['short_sell_orders'].shift(order_shift)
    data['long_sell_orders'] = data['long_sell_orders'].shift(order_shift)
    data['short_buy_orders'] = data['short_buy_orders'].shift(order_shift)
    data['maxima'] = data['maxima'].shift(order_shift)
    data['minima'] = data['minima'].shift(order_shift)
    data['>zero'] = data['>zero'].shift(order_shift)
    data['<zero'] = data['<zero'].shift(order_shift)

    data['immobilized_long'] = 0
    data['immobilized_short'] = 0

    long_orders = pd.DataFrame()
    for date, value in data['long_buy_orders'].iteritems():
        if data.loc[date, 'long_buy_orders'] == 2:
            to_invest = start_cash if len(long_orders) == 0 else long_orders.iloc[-1]['money']

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
                if remaining_funds.loc[date:sell_date].min() >= to_invest:
                    sell_date = min(abort_date, sell_date)
                    data.loc[sell_date, 'long_sell_orders'] = 2
                    data['long_buy_orders'][list(data['minima'].loc[date_plus_1:sell_date].dropna().index)] = 1
                    data.loc[date:sell_date, 'immobilized_long'] += 1

                    buy_price = data['open'][date] * (1 + fee)
                    sell_price = data['open'][sell_date] * (1 - fee)
                    roi = sell_price/buy_price
                    new_money = roi * to_invest
                    profit = new_money - to_invest
                    trade = pd.DataFrame([date, sell_date, buy_price, sell_price, roi, new_money, profit]).T
                    trade.columns = ['buy_date', 'sell_date', 'buy_price', 'sell_price', 'roi', 'money', 'profit']
                    long_orders = long_orders.append(trade).reset_index(drop=True)

                    remaining_funds.loc[date:sell_date] -= to_invest  # Simplified, maybe add a modulo someday
                    remaining_funds.loc[sell_date:] += profit
                    invested_funds.loc[date:sell_date] += to_invest

                    try:
                        orders.loc[date] += 1
                        orders.loc[sell_date] += 1
                    except:
                        pass
                else:
                    data.loc[date, 'long_buy_orders'] = 0
            else:
                data.loc[date:, 'long_buy_orders'] = np.nan


    short_orders = pd.DataFrame()
    for date, value in data['short_sell_orders'].iteritems():
        if data['short_sell_orders'].loc[date] == 2:
            to_invest = start_cash if len(short_orders) == 0 else short_orders.iloc[-1]['money']

            date_plus_1 = int(datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(minutes=period), '%Y%m%d%H%M'))
            abort_date = data.index[-1]

            if sell_trigger_short == 'extremum':
                cover_date = data['minima'].loc[date_plus_1:].first_valid_index()
            elif sell_trigger_short == 'zero crossing':
                cover_date = data['<zero'].loc[date_plus_1:].first_valid_index()

            if data.index.get_loc(date) + 3*390//period < len(data):
                date_plus_3 = int(datetime.strftime(datetime.strptime(str(date), '%Y%m%d%H%M') + timedelta(days=min_days_before_abort), '%Y%m%d%H%M'))
                if (data['diff'].loc[date_plus_3:] > data['maxima'].loc[date]).any():
                    abort_date = (data['diff'].loc[date_plus_3:] > data['maxima'].loc[date]).idxmax()

            if cover_date is not None:
                if remaining_funds.loc[date:cover_date].min() >= to_invest:
                    cover_date = min(abort_date, cover_date)
                    data.loc[cover_date, 'short_buy_orders'] = 2
                    data['short_sell_orders'][list(data['maxima'].loc[date_plus_1:cover_date].dropna().index)] = 1
                    data.loc[date:cover_date, 'immobilized_short'] += 1

                    cover_price = data['open'][cover_date] * (1 + fee)
                    sell_price = data['open'][date] * (1 - fee)
                    new_money = to_invest / sell_price * (sell_price - cover_price) + to_invest
                    roi = new_money / to_invest
                    profit = new_money - to_invest

                    trade = pd.DataFrame([date, cover_date, cover_price, sell_price, roi, new_money, profit]).T
                    trade.columns = ['sell_date', 'cover_date', 'cover_price', 'sell_price', 'roi', 'money', 'profit']
                    short_orders = short_orders.append(trade).reset_index(drop=True)

                    remaining_funds.loc[date:cover_date] -= to_invest  # Simplified, maybe add a modulo someday
                    remaining_funds.loc[cover_date:] += profit
                    invested_funds.loc[date:cover_date] += to_invest

                    try:
                        orders.loc[date] += 1
                        orders.loc[sell_date] += 1
                    except:
                        pass
                else:
                    data.loc[date, 'short_sell_orders'] = 0
            else:
                data.loc[date:, 'short_sell_orders'] = np.nan

    long_profit, short_profit = 0, 0
    if len(long_orders):
        long_profit = long_orders['profit'].sum()
        print('Profit on long :', int(long_profit))
    if len(short_orders):
        short_profit = short_orders['profit'].sum()
        print('Profit on short :', int(short_profit))

    profit = long_profit + short_profit
    print('TOTAL :', int(profit))

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
    return profit


# Change this to the path where your pickled stocks data is
data_path = 'PickledStocksData'

# Simulation parameters
period = 5
start_cash = 65000
fee = 0  # Approximation
budget = 5000000
order_threshold = 0.25
verif_size = 1
order_shift = 5
extremums_order = 5
min_days_before_abort = 5
sell_trigger_long = 'zero crossing'  # 'zero crossing' or 'extremum'
sell_trigger_short = 'zero crossing'
plot = False

results = []
remaining_funds, invested_funds, orders = None, None, None
tickers = [file for file in os.listdir(data_path) if '_'+str(period)+'.' in file]
shuffle(tickers)
tickers = tickers[:300]
# tickers = ['INFO_5.p', 'AMZN_5.p', 'AAPL_5.p', 'MSFT_5.p', 'FB_5.p', 'GOOG_5.p', 'TSLA_5.p']
for stock in tickers:
    print('\n' + stock.split('.p')[0])
    print('{}/{}'.format(tickers.index(stock)+1, len(tickers)))
    data = pickle.load(open(data_path + "/{}".format(stock), "rb"))
    remaining_funds = data['open'] * 0 + budget if remaining_funds is None else remaining_funds
    invested_funds = data['open'] * 0 if invested_funds is None else invested_funds
    orders = data['open'] * 0 if orders is None else orders
    if len(data) > 390 // period * 26 and stock not in []:
        yearly_profit = simulate(data, period, fee, start_cash, remaining_funds, invested_funds, orders, order_threshold, verif_size, order_shift, extremums_order, min_days_before_abort, sell_trigger_long, sell_trigger_long, plot=plot, allow_short=True)
        remaining_funds = remaining_funds.fillna(method='ffill')
        invested_funds = invested_funds.fillna(method='ffill')
        results.append((stock.split('.p')[0], yearly_profit))

results = pd.DataFrame(results, columns=['Stock', 'yearly_profit']).set_index('Stock')
results = results.sort_values('yearly_profit', ascending=False)
results.to_csv('results2.csv')
print(results)
print(results.describe())

total_profit = round(results.loc[:, 'yearly_profit'].sum(), 1)
print('Total profit :', total_profit)
print('Max immobilized funds :', budget)
print('Interest rate over period :', round(100*total_profit/budget, 1), '%')

report = pd.DataFrame([period, fee, start_cash, order_threshold, verif_size, order_shift, extremums_order, min_days_before_abort, sell_trigger_long, sell_trigger_short, total_profit, budget, round(100*total_profit/budget, 1)]).T
report.columns = ['period', 'fee', 'start_cash', 'order_threshold', 'verif_size', 'order_shift', 'extremums_order', 'min_days_before_abort', 'sell_trigger_long', 'sell_trigger_short', 'total_profit', 'max_immo', 'interest']
report_old = pd.read_csv('report.csv')
report = report_old.append(report, ignore_index=True)
report.to_csv('report.csv', index=False)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
remaining_funds.name = 'Cash'

remaining_funds.reset_index(drop=True).plot(ax=ax1, legend=' ')
invested_funds.name = 'Invested'
invested_funds.reset_index(drop=True).plot(ax=ax1, legend=' ')
net_worth = remaining_funds.add(invested_funds).reset_index(drop=True)
net_worth.name = 'Net worth'
net_worth.plot(ax=ax1, legend=' ')
ax1.set_title('Money Evolution')

orders.reset_index(drop=True).plot(ax=ax2)
ax2.set_title('Number of trades per period (5 sec here)')
plt.show()