import pickle

data = pickle.load(open('FullData_5.p', 'rb'))
print(data.index[0], data.index[-1])
print(len(data) // (390/5) // 22)