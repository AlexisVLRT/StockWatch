from iexfinance import Stock
from datetime import datetime
import pandas as pd
import Database_credentials as dc
import mysql.connector
from threading import Thread
import time


def market_is_open():
    worked_days = [0, 5]
    worked_hours = [9.5, 16]
    if worked_days[0] < datetime.now().weekday() < worked_days[1] and worked_hours[0] * 60 < (datetime.now().hour - 5) * 60 + datetime.now().minute < worked_hours[1] * 60:
        return True
    return False


def fetch_market_data(tickers):
    prices = {}
    for i in range(0, len(tickers), 99):
        # print('{}/{}'.format(i, len(tickers)))
        stock = Stock(tickers[i:i + 99])
        batch_prices = {ticker: (datetime.now(), value) for ticker, value in stock.get_price().items()}
        prices.update(batch_prices)
    batch_prices = {ticker: (datetime.now(), value) for ticker, value in Stock(tickers[-(len(tickers) % 99):]).get_price().items()}
    prices.update(batch_prices)
    return prices


def upload_data(data_as_dict):
    conn = mysql.connector.connect(host=dc.host, user=dc.user, password=dc.password, database=dc.database)
    cursor = conn.cursor()
    for ticker, (timestamp, price) in data_as_dict.items():
        cursor.execute("""
            INSERT INTO `{ticker}` (Time, Price) VALUES ('{timestamp}','{price}')
            """.format(ticker=ticker.replace('.', 'dot'), timestamp=timestamp, price=price))
        conn.commit()
    cursor.close()
    conn.close()


def fetch_and_upload(tickers):
    data = fetch_market_data(tickers)
    upload_data(data)
    print('Done')


tickers = list(pd.read_csv('MostTraded.csv', header=-1).iloc[:, 0])
previous_minute = datetime.now().minute
while 1:
    if market_is_open() and datetime.now().minute != previous_minute:
        print('Fetching data')
        t = Thread(target=fetch_and_upload, args=(tickers,))
        t.start()
        previous_minute = datetime.now().minute
    time.sleep(1)