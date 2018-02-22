'''
using gridsearch method to combine data
'''
import numpy as np
from pandas import read_csv
import pandas as pd

def err(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
dataframe = read_csv('../file/mix/2122purchase.csv', engine='python')
dataset = dataframe.values
actual_redeem = dataset[:,1]
lstm = dataset[:,2]
arima_with_mean = dataset[:,3]
actual_redeem_array = np.array(actual_redeem)
lstm_array = np.array(lstm)
arima_with_mean_array = np.array(arima_with_mean)
# actual_redeem_array_times = actual_redeem_array*0.2
range_ten = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
arr = np.linspace(0,1,100)
result = []
err1 = err(actual_redeem,lstm)
err2 = err(actual_redeem,arima_with_mean)
print(err1)
print(err2)
# a = float(arima_with_mean_array)*(1-0.1)
for i in arr:
    # print(type(i))
    res = lstm_array*i+(arima_with_mean_array)*(1.0-i)
    err_single = err(actual_redeem_array,res)
    result.append(err_single)
for i in result:
    min_index = result.index(min(result))
mix_data = pd.DataFrame({'0218_mix_index':arr,'mix_data':result})
mix_data.to_csv("0218_mix_data_pur.csv")
print("arr[min_index]:",arr[min_index])
print("result[min_index]",result[min_index])
print(min_index)
print(result)
print(111)