import iexfinance
import pandas as pd
import datetime

all_symbols = [stock['symbol'] for stock in iexfinance.get_available_symbols()]
volumes = {}

data = iexfinance.get_historical_data('FB', datetime.datetime(2018, 4, 29), datetime.datetime(2018, 4, 30))
volume = list(list(data.values())[0].values())[0]['volume']
iexfinance.Stock('FB')

for i in range(0, len(all_symbols), 99):
    print('{}/{}'.format(i, len(all_symbols)))
    stock = iexfinance.Stock(all_symbols[i:i+99])
    volumes.update(stock.get_volume())

volumes.update(iexfinance.Stock(all_symbols[-(len(all_symbols) % 99):]).get_volume())

volumes = pd.Series(volumes)
volumes.sort_values(inplace=True, ascending=False)
volumes = volumes.iloc[:1000]
volumes.to_csv('out.csv')