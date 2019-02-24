import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn;seaborn.set()
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
from random import shuffle
import DataBase
from threading import Thread
from multiprocessing import Manager, Pool, Queue, Process
import time


class RealTime:
    def __init__(self, db_id, n_cores, tickers_pairs, cash, cash_per_trade,  offset, queue):
        print(tickers_pairs)
        tickers_pairs = tickers_pairs[~(tickers_pairs['Stock A'].str.contains('HMNY') | tickers_pairs['Stock B'].str.contains('HMNY'))]
        self.db_id = db_id
        self.n_cores = n_cores
        self.tickers_pairs = tickers_pairs.reset_index(drop=True)
        self.queue = queue
        self.period = 5
        self.cash = cash / n_cores
        self.cash_per_trade = cash_per_trade

        self.offset = offset
        self.past_data = pickle.load(open("FullData_5.p", "rb")).fillna(method='bfill').fillna(method='ffill')
        self.past_data = self.past_data[~self.past_data.index.duplicated(keep='last')]

        self.positions, self.invested = DataBase.get_open_positions(self.db_id)
        print(self.invested, self.cash)

        self.start_simulation()

    def start_simulation(self):
        print(self.offset, int(252 / 12 * 1 * (390 / self.period)) + self.offset)
        for i in range(int(252 / 12 * 1 * (390 / self.period))):
            cash, invested = self.get_indicators(i + self.offset)
            self.queue.put((i, cash, invested))

    def get_indicators(self, i):
        three_months = int(252 / 12 * 3 * (390 / self.period))
        roi = self.past_data.iloc[i + three_months: i + 2 * three_months]

        pairs_traded = set(np.array(self.positions.loc[:, 'TradeID'].str.split('|').values.tolist())[:, 0]) if len(self.positions) else []
        for _, pair in self.tickers_pairs.iterrows():
            stock_a = roi.loc[:, pair[0]]
            stock_b = roi.loc[:, pair[1]]

            gamma, intercept = np.linalg.lstsq(np.vstack([stock_b.values, np.ones(len(stock_b.values))]).T, stock_a.values, rcond=None)[0]
            mu = (stock_b - stock_a * gamma).mean()
            residual = (stock_b - stock_a * gamma) - mu

            norm_residual = pd.Series(StandardScaler(with_std=True).fit_transform(residual.values.reshape(-1, 1)).reshape(1, -1)[0])

            if '{}/{}'.format(pair[0], pair[1]) not in pairs_traded and self.cash > 2 * self.cash_per_trade:
                if norm_residual.iloc[-2] > 2 > norm_residual.iloc[-1]:
                    # Long A Short B
                    to_invest = 0.02 * (self.cash + self.invested)
                    self.invested += to_invest
                    self.cash -= to_invest

                    print('Buying ', pair[0], i)
                    self.open_position(pair[0], entry_date=stock_a.index[-1], position=1, entry_price=stock_a.iloc[-1], entry_money=0.02 * (self.cash + self.invested), ticker_pair=pair)
                    # print('Shorting ', pair[1], i)
                    # self.open_position(pair[1], entry_date=stock_b.index[-1], position=-1, entry_price=stock_b.iloc[-1], entry_money=self.cash_per_trade, ticker_pair=pair)

            elif '{}/{}'.format(pair[0], pair[1]) not in pairs_traded and self.cash > 2 * self.cash_per_trade:
                if norm_residual.iloc[-2] < -2 < norm_residual.iloc[-1]:
                    # Short A Long B
                    to_invest = 0.02 * (self.cash + self.invested)
                    self.invested += to_invest
                    self.cash -= to_invest

                    print('Buying ', pair[1], i)
                    self.open_position(pair[1], entry_date=stock_b.index[-1], position=1, entry_price=stock_b.iloc[-1], entry_money=0.02 * (self.cash + self.invested), ticker_pair=pair)
                    # print('Shorting ', pair[0], i)
                    # self.open_position(pair[0], entry_date=stock_a.index[-1], position=-1, entry_price=stock_a.iloc[-1], entry_money=self.cash_per_trade, ticker_pair=pair)

        for pair in pairs_traded:
            stock_a = roi.loc[:, pair.split('/')[0]]
            stock_b = roi.loc[:, pair.split('/')[1]]

            gamma, intercept = np.linalg.lstsq(np.vstack([stock_b.values, np.ones(len(stock_b.values))]).T, stock_a.values, rcond=None)[0]
            mu = (stock_b - stock_a * gamma).mean()
            residual = (stock_b - stock_a * gamma) - mu

            norm_residual = pd.Series(StandardScaler(with_std=True).fit_transform(residual.values.reshape(-1, 1)).reshape(1, -1)[0])

            if (norm_residual.iloc[-2] > 0 > norm_residual.iloc[-1]) or (norm_residual.iloc[-2] < 0 < norm_residual.iloc[-1]) or norm_residual.iloc[-2] > -4 > norm_residual.iloc[-1] or norm_residual.iloc[-2] < 4 < norm_residual.iloc[-1]:
                trades_of_pair = self.positions[self.positions['TradeID'].str.contains(pair)]
                trade_long = trades_of_pair[trades_of_pair['Position'] == 1]
                ticker_long = trade_long['Ticker'].values[0]
                stock_long = roi.loc[:, ticker_long]
                new_money = (stock_long.iloc[-1] / trade_long['EntryPrice'] * trade_long['EntryMoney']).values[0]
                self.invested -= trade_long['EntryMoney'].values[0]
                self.cash += new_money
                print('Selling ', ticker_long, i)
                self.close_position(ticker=ticker_long, trade_id=trade_long['TradeID'].values[0], money_in=trade_long['EntryMoney'].values[0], exit_date=stock_long.index[-1], position=1, exit_price=stock_long.iloc[-1], exit_money=new_money)

                # trade_short = trades_of_pair[trades_of_pair['Position'] == -1]
                # print(trade_short)
                # ticker_short = trade_short['Ticker'].values[0]
                # print(ticker_short)
                # stock_short = roi.loc[:, ticker_short]
                # new_money = (trade_short['EntryMoney'] / trade_short['EntryPrice'] * (trade_short['EntryPrice'] - stock_short.iloc[-1]) + trade_short['EntryMoney']).values[0]
                # self.invested -= trade_short['EntryMoney'].values[0]
                # self.cash += new_money
                # print('Covering ', ticker_short, i)
                # self.close_position(ticker=ticker_short, trade_id=trade_short['TradeID'].values[0], money_in=trade_short['EntryMoney'].values[0], exit_date=stock_short.index[-1], position=1, exit_price=stock_short.iloc[-1], exit_money=new_money)

        return self.cash, self.invested

    def ingest(self, data_line):
        self.past_data = self.past_data.append(data_line)

    def open_position(self, ticker, entry_date, position, entry_price, entry_money, ticker_pair=None):
        if ticker_pair is None:
            trade_id = '{}|{}|{}|{}|{}'.format(ticker, entry_date, position, entry_price, entry_money)
        else:
            trade_id = '{}/{}|{}|{}|{}|{}'.format(ticker_pair[0], ticker_pair[1], entry_date, position, entry_price, entry_money)

        self.positions = self.positions.append(pd.DataFrame([[ticker, position, entry_date, entry_price, entry_money, trade_id]], columns=['Ticker', 'Position', 'EntryDate', 'EntryPrice', 'EntryMoney', 'TradeID']), sort=False)
        self.positions = self.positions.reset_index(drop=True)
        Thread(target=DataBase.open_position, args=(self.db_id, trade_id, ticker, position, entry_date, entry_price, entry_money)).start()

    def close_position(self, ticker, trade_id, money_in, exit_date, position, exit_price, exit_money):
        assert(position == -1 or position == 1)

        self.positions = self.positions[~(self.positions['TradeID'].str.contains(trade_id.split('|')[0]))]

        profit = exit_money / money_in
        Thread(target=DataBase.close_position, args=(self.db_id, ticker, trade_id, exit_date, exit_price, exit_money, profit)).start()


if __name__ == '__main__':
    db_id = 2
    n_cores = 1
    n_stocks = 100
    cash = 5000000
    cash_per_trade = 50000
    period = 5

    pool = Pool(processes=1)
    q = Queue()

    money = pd.DataFrame([[cash, 0, cash]], columns=['Cash', 'Invested', 'Net worth'])
    for month in range(0, 11):
        offset = int(252 / 12 * month * (390 / period))

        pairs = pickle.load(open('adf_scores{}.p'.format(month), 'rb')).sort_values(by='Score').loc[:, ['Stock A', 'Stock B']][:n_stocks]
        pairs = [pairs[int(i * len(pairs) / n_cores):int((i + 1) * len(pairs) / n_cores)] for i in range(n_cores)]

        for i in range(n_cores):
            # RealTime(db_id, n_cores, pairs[0], cash, cash_per_trade, offset, q)
            Process(target=RealTime, args=(db_id, n_cores, pairs[0], cash, cash_per_trade, offset, q)).start()

        last_point = time.time()
        plot, index = False, 0
        while index < int(252 / 12 * 1 * (390 / period))-1:
            index, cash, invested = q.get()
            print(index)
            money = money.append(pd.DataFrame([[cash, invested, cash + invested]], columns=['Cash', 'Invested', 'Net worth'])).reset_index(drop=True)
            money.to_csv('SimulationResults/PairsTrading.csv')
            # print(time.time()-last_point)
            last_point = time.time()
            plt.clf()
            plt.plot(money)
            plt.pause(0.000001)
    plt.clf()
    plt.plot(money)
    plt.show()
