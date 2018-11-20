import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn;seaborn.set()
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import os
from iexfinance import Stock


stock = 'MRTX'
data_path = 'PickledStocksData'
period = 5

data = pd.read_csv('recap.csv', header=-1, index_col=1)
data = data.dropna()
data = data.loc[stock, :]

refined = pd.DataFrame(data[3])
refined.columns = ['Entry Date']
refined['Exit Date'] = data[6]
refined['Position'] = data[2]
refined['MoneyIn'] = data[5]
refined['MoneyOut'] = data[8]
refined['MoneyDelta'] = refined['MoneyOut'] - refined['MoneyIn']


hist_data = pd.DataFrame(pickle.load(open(data_path + "/{}_5.p".format(stock), "rb"))['open'])
hist_data = hist_data[~hist_data.index.duplicated(keep='last')].iloc[:11750]
hist_data['Order'] = pd.Series()


ema24 = hist_data.iloc[:, 0].ewm(span=26 * 390 // period).mean()
ema12 = hist_data.iloc[:, 0].ewm(span=12 * 390 // period).mean()
macd = ema12 - ema24
signal = macd.ewm(span=9 * 390 // period).mean()
diff = (macd - signal)
diff = pd.Series(StandardScaler(with_mean=False).fit_transform(diff.values.reshape(-1, 1)).flatten())
diff2 = (diff.fillna(0).diff().ewm(span=0.5 * 390 // period).mean()).fillna(0)*200
# diff2 = pd.Series(StandardScaler(with_mean=False).fit_transform(diff2.values.reshape(-1, 1)).flatten())

for ticker, (entry_date, exit_date, position, money_in, money_out, money_delta) in refined.iterrows():
    if position == 1:
        # Long
        hist_data.loc[entry_date, 'Order'] = 0  # Buy
        hist_data.loc[exit_date, 'Order'] = 1  # Sell
    if position == -1:
        # Short
        hist_data.loc[entry_date, 'Order'] = 2  # Short
        hist_data.loc[exit_date, 'Order'] = 3  # Cover

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')

ax2.fill_between(list(diff.reset_index(drop=True).index), diff.fillna(0).values, alpha=0.5)
ax2.fill_between(list(diff.reset_index(drop=True).index), diff2.fillna(0).values, alpha=0.5)
hist_data['open'].reset_index(drop=True).plot(ax=ax1)

for order in hist_data.dropna().iterrows():
    date, value = order[0], int(order[1]['Order'])
    colors = ['blue', 'cyan', 'red', 'orange']
    ax1.axvline(x=list(hist_data.index).index(date), color=colors[value], linewidth=0.5)
    ax2.axvline(x=list(hist_data.index).index(date), color=colors[value], linewidth=0.5)
    # ax3.axvline(x=list(hist_data.index).index(date), color=colors[value], linewidth=0.5)

plt.show()
