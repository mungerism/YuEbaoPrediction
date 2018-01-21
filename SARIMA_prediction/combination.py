
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras.models
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential

import pandas as pd
from sklearn.externals import joblib
import os
def err(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
dataframe = read_csv('../file/Gridsearchtask3.csv', engine='python')
dataset = dataframe.values
actual_redeem = dataset[:,1]
lstm = dataset[:,2]
arima_with_mean = dataset[:,3]
actual_redeem_array = np.array(actual_redeem)
lstm_array = np.array(lstm)
arima_with_mean_array = np.array(arima_with_mean)
# actual_redeem_array_times = actual_redeem_array*0.2
range_ten = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
result = []
# a = float(arima_with_mean_array)*(1-0.1)
for i in range_ten:
    # print(type(i))
    res = lstm_array*i+(arima_with_mean_array)*(1.0-i)
    err_single = err(actual_redeem_array[277:],res[277:])
    result.append(err_single)
for i in result:
    min_index = result.index(min(result))
print(range_ten[min_index])
print(result)
print(111)