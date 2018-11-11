from datetime import datetime
import pandas as pd
import pickle
import os
import time


def parse(period):
    most_volatile = list(pd.read_csv('mostVolatile.csv', header=-1).iloc[:, 0])
    source_files = os.listdir('FullData')
    source_files = [file for file in source_files if '_'+str(period) in file]
    pickle_files = os.listdir('PickledData')
    for file in pickle_files:
        if '_' + str(period) in file:
            os.remove('PickledData/' + file)

    for file in source_files:
        start = time.time()
        print('{}/{}'.format(source_files.index(file)+1, len(source_files)))
        pickle_files = os.listdir('D:/PickledStocksData')
        if '_' + str(period) + '.' in file:
            data = pd.read_csv('FullData/' + file, header=-1)
            data.columns = ['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            data = data.drop(columns=['high', 'low', 'close'])
            # data['datetime'] = data['datetime'].apply(lambda date: datetime.strptime(str(date), '%Y%m%d%H%M'))
            data = data[data['ticker'].isin(most_volatile)]
            tickers = set(data['ticker'])
            print(len(tickers))
            for ticker in tickers:
                stock = data[data['ticker'] == ticker].set_index('datetime')
                if ticker + '_' + str(period) + '.p' in pickle_files:
                    previous_stock_data = pickle.load(open('D:/PickledStocksData/{}_{}.p'.format(ticker, period), 'rb'))
                    stock = stock.append(previous_stock_data)
                    stock = stock.sort_index()
                pickle.dump(stock, open('D:/PickledStocksData/{}_{}.p'.format(ticker, period), 'wb'))
        print(round(time.time()-start, 1))

period = 5
parse(period)