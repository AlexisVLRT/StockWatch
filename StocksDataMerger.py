import pandas as pd
import numpy as np
import pickle
import os

data_path = 'PickledStocksData'
period = 5

tickers = [file for file in os.listdir(data_path) if '_'+str(period)+'.' in file][:2]
tickers_names = [ticker.replace('_{}.p'.format(period), '') for ticker in tickers]
data = pickle.load(open(data_path + "/{}".format(tickers[0]), "rb"))

index = pd.MultiIndex.from_product([list(data.index), tickers_names], names=['date', 'ticker'])
frame = pd.DataFrame(np.random.randn(len(list(data.index)) * len(tickers_names), 1), index=index)
print(frame.loc[:])

# for stock in tickers:
#     data = pickle.load(open(data_path + "/{}".format(stock), "rb"))
#     data = data.loc[:, 'open']
#     print(data)
#     frame.loc[(:, stock), :] = data
