import iexfinance
import pandas as pd
import mysql.connector
import Database_credentials as dc

tickers = list(pd.read_csv('MostTraded.csv', header=-1).iloc[:, 0])
print(tickers)

conn = mysql.connector.connect(host=dc.host, user=dc.user, password=dc.password, database=dc.database)
cursor = conn.cursor()


for ticker in tickers:
    print('{}/{}'.format(tickers.index(ticker), len(tickers)), ticker)
    cursor.execute("""
            CREATE TABLE `{ticker}`
            (
                Time TIMESTAMP DEFAULT current_timestamp PRIMARY KEY,
                Price FLOAT
            )""".format(ticker=ticker.replace('.', 'dot')))
    cursor.execute("""CREATE UNIQUE INDEX {ticker}_Time_uindex ON `{ticker}` (Time)""".format(ticker=ticker.replace('.', 'dot')))
    conn.commit()
cursor.close()
conn.close()