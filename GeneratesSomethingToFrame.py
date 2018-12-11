import pandas as pd
from matplotlib import pyplot as plt
import pickle
import statsmodels.tsa.stattools as ts

data = pickle.load(open("FullData_5.p", "rb"))[:10000]
data = data.fillna(method='bfill')
data = data.fillna(method='ffill')
data = data.iloc[::78, :]
data = data.dropna(axis=1, how='all')

results = pd.DataFrame()
for a1 in data.columns:
    print(a1)
    print(str(list(data.columns).index(a1)) + '/' + str(len(data.columns)))
    for a2 in data.columns:
        if a1 != a2:
            test_result = ts.coint(data[a1], data[a2])
            results.loc[a1, a2] = test_result[1]

results.to_csv('CoInt_mat.csv')
plt.imshow(results)
plt.show()
