import pandas as pd
import numpy as np
import pickle
import os

data_path = 'D:/PickledStocksData'
period = 5

tickers = [file for file in os.listdir(data_path) if '_'+str(period)+'.' in file]
tickers_names = [ticker.replace('_{}.p'.format(period), '') for ticker in tickers]
full_data = pd.DataFrame()

for stock in tickers:
    data = pickle.load(open(data_path + "/{}".format(stock), "rb"))['open']
    data = data[~data.index.duplicated(keep='last')]
    full_data[stock.replace('_{}.p'.format(period), '')] = data
pickle.dump(full_data, open('FullData_{}.p'.format(period), 'wb'))
print(full_data)
