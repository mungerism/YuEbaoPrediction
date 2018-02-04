'''
using gridsearch method to combine data
'''
import numpy as np
from pandas import read_csv
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
def err(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
dataframe = read_csv('../file/final_data/mix/redeem/arima_lstm.csv', engine='python')
dataset = dataframe.values
label = dataset[:,1]
lstm = dataset[:,2]
arima = dataset[:,3]
regr = linear_model.LinearRegression()

f = [lstm,arima]
f = np.transpose(f)
# lr = LogisticRegression()
regr.fit(f, label)
# lr.fit(f, label)
final = regr.predict(f)
err = err(label[270:],final[270:])
print(err)
print(222)