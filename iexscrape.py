import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import time
from threading import Thread


def get_data(ticker, date: datetime):
    date_str = date.strftime("%Y%m%d")
    r = requests.get('https://api.iextrading.com/1.0/stock/{ticker}/chart/date/{date}'.format(ticker=ticker.lower(), date=date_str))
    data = pd.DataFrame(r.json())
    if len(data):
        data = data[['marketHigh', 'date', 'minute', 'volume']]
        data['datetime'] = data["date"].map(str) + ' ' + data["minute"]
        data = data.drop(columns=['date', 'minute'])
        data['datetime'] = data['datetime'].apply(lambda date: datetime.strptime(date, '%Y%m%d %H:%M'))
        data = data.replace(-1, np.nan)
        data['marketHigh'] = data['marketHigh'].interpolate()
        # data = data[data['average'] != -1]
        data = data.set_index('datetime')
    return data


def get_historical(ticker, day_count, end_date: datetime):
    data = pd.DataFrame()
    for date in (end_date - timedelta(day_count - n) for n in range(day_count)):
        print(date)
        day_data = get_data(ticker, date)
        data = data.append(day_data)
    return data


def create_pickle(ticker):
    print(ticker)
    hist_data = get_historical(ticker, 60, datetime.now())
    pickle.dump(hist_data, open('stocksData/' + ticker + '.p', 'wb'))


tickers = list(pd.read_csv('mostVolatile.csv', header=-1)[0])
batch_size = 50
for i in range(0, len(tickers), batch_size):
    threads = []
    for ticker in tickers[i:i+batch_size]:
        t = Thread(target=create_pickle, args=(ticker, ))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()