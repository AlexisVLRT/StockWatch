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
from multiprocessing import Manager, Pool
import time
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


stocks_data = pickle.load(open("FullData_5.p", "rb"))
stocks_data = stocks_data.iloc[:, :100]
three_months = int(252 / 12 * 3 * (390 / 5))

for i in range(10, 11):
    offset = int(252 / 12 * i * (390 / 5))
    stocks_data_ds = stocks_data.iloc[offset:offset+three_months:6, :]
    print(stocks_data_ds)
    correlated_stocks = pd.DataFrame(columns=['Stock A', 'Stock B', 'Score'])
    for ticker_a in range(len(stocks_data.columns)):
        print('{}/{}'.format(ticker_a, len(stocks_data.columns)))
        for ticker_b in range(ticker_a+1, len(stocks_data.columns)):
            ser_a = stocks_data_ds.iloc[:, ticker_a] - stocks_data_ds.iloc[:, ticker_a].mean()
            ser_b = stocks_data_ds.iloc[:, ticker_b] - stocks_data_ds.iloc[:, ticker_b].mean()
            num = (ser_a * ser_b).sum()
            den = ((ser_a ** 2).sum() * (ser_b ** 2).sum()) ** 0.5
            corr = num / den
            correlated_stocks = correlated_stocks.append(pd.DataFrame([[list(stocks_data.columns)[ticker_a], list(stocks_data.columns)[ticker_b], corr]], columns=['Stock A', 'Stock B', 'Score']))

    correlated_stocks = correlated_stocks.sort_values(by='Score', inplace=False, ascending=False)
    most_correlated = correlated_stocks.iloc[:200, :]
    adf_scores = pd.DataFrame(columns=['Stock A', 'Stock B', 'Score'])
    i = 0
    for pair in most_correlated.iterrows():
        print('{}/{}'.format(i, len(most_correlated)))
        i += 1
        stock_a = pair[1].loc['Stock A']
        stock_b = pair[1].loc['Stock B']
        X = stocks_data.loc[:, stock_a].fillna(method='bfill').fillna(method='ffill')
        Y = stocks_data.loc[:, stock_b].fillna(method='bfill').fillna(method='ffill')
        Xsm = sm.add_constant(X.values)
        model = sm.OLS(Y.values, Xsm)
        intercept, gamma = model.fit().params
        mu = (Y - X * gamma).mean()
        residual = (Y - X * gamma) - mu
        adf = adfuller(residual)[0]
        adf_scores = adf_scores.append(pd.DataFrame([[stock_a, stock_b, adf]], columns=['Stock A', 'Stock B', 'Score']))

    adf_scores = adf_scores.sort_values('Score')
    pickle.dump(adf_scores, open('adf_scores{}.p'.format(i), 'wb'))
