import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn;seaborn.set()
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import os
from iexfinance import Stock


stock = 'ADI'
data_path = 'PickledStocksData'
period = 5

data = pd.read_csv('AllMovesOptiMACD.csv', header=-1, index_col=1)
data = data.dropna()
data = data.loc[stock, :]

print(data)
refined = pd.DataFrame(data[3])
refined.columns = ['Entry Date']
refined['Exit Date'] = data[6]
refined['Position'] = data[2]
refined['MoneyIn'] = data[5]
refined['MoneyOut'] = data[8]
refined['MoneyDelta'] = refined['MoneyOut'] - refined['MoneyIn']


hist_data = pd.DataFrame(pickle.load(open(data_path + "/{}_5.p".format(stock), "rb"))['open'])
hist_data = hist_data[~hist_data.index.duplicated(keep='last')].iloc[:]
hist_data['Order'] = pd.Series()


ema24 = hist_data.iloc[:, 0].ewm(span=26 * 390 // period * 0.5).mean()
ema12 = hist_data.iloc[:, 0].ewm(span=12 * 390 // period * 0.5).mean()
macd = ema12 - ema24
signal = macd.ewm(span=9 * 390 // period * 0.5).mean()
diff = (macd - signal)
diff = pd.Series(StandardScaler(with_mean=False).fit_transform(diff.values.reshape(-1, 1)).flatten())

ema242 = hist_data.iloc[:, 0].ewm(span=26 * 390 // period * 2).mean()
ema122 = hist_data.iloc[:, 0].ewm(span=12 * 390 // period * 2).mean()
macd2 = ema122 - ema242
signal2 = macd2.ewm(span=9 * 390 // period * 2).mean()
diff2 = (macd2 - signal2)
diff2 = pd.Series(StandardScaler(with_mean=False).fit_transform(diff2.values.reshape(-1, 1)).flatten())

ema243 = hist_data.iloc[:, 0].ewm(span=26 * 390 // period).mean()
ema123 = hist_data.iloc[:, 0].ewm(span=12 * 390 // period).mean()
macd3 = ema123 - ema243
signal3 = macd3.ewm(span=9 * 390 // period).mean()
diff3 = (macd3 - signal3)
diff3 = pd.Series(StandardScaler(with_mean=False).fit_transform(diff3.values.reshape(-1, 1)).flatten())
diff_ = (diff3.fillna(0).diff().ewm(span=0.5 * 390 // period).mean()).fillna(0) * 200

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

ax2.plot(list(diff2.reset_index(drop=True).index), macd3)
ax2.plot(list(diff2.reset_index(drop=True).index), signal3)
ax2.fill_between(list(diff2.reset_index(drop=True).index), diff3.fillna(0).values, alpha=0.5)
# ax2.fill_between(list(diff.reset_index(drop=True).index), diff_.fillna(0).values, alpha=0.5)
# ax2.fill_between(list(diff.reset_index(drop=True).index), diff3.fillna(0).values, alpha=0.5)
hist_data['open'].reset_index(drop=True).plot(ax=ax1)

price_in = hist_data.iloc[0, 0]
y_profit = 0.5
a = -y_profit * period * price_in / (390 * 250)
x = np.array(list(diff2.reset_index(drop=True).index))
# ax1.plot(a*x + price_in - price_in*0.01, color='yellow', linewidth=0.5, linestyle='--')
# ax1.plot(a*(-x) + price_in + price_in*0.01, color='blue', linewidth=0.5, linestyle='--')

save = pd.DataFrame([hist_data.index, macd3.fillna(0), diff3.fillna(0)]).T.set_index(0, drop=True)
save.columns = ['MACD', 'Signal']
save['trigger'] = np.nan
for order in hist_data.dropna().iterrows():
    date, value = order[0], int(order[1]['Order'])
    colors = ['blue', 'cyan', 'red', 'orange']
    save.loc[date, 'Trigger'] = ['buy', 'sell', 'short', 'cover'][int(order[1]['Order'])]
    ax1.axvline(x=list(hist_data.index).index(date), color=colors[value], linewidth=0.5)
    ax2.axvline(x=list(hist_data.index).index(date), color=colors[value], linewidth=0.5)
    # ax3.axvline(x=list(hist_data.index).index(date), color=colors[value], linewidth=0.5)

save.to_csv('SimulationResults/MACDexample$ADI.csv')
plt.show()
