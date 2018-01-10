# from sklearn.preprocessing import MinMaxScaler
# import numpy
# scaler = MinMaxScaler(feature_range=(0, 1))
# a = [2,23,4,5,2,1]
# arr = numpy.array(a)
# # mm =arr.reshape(-1,1)
# mm = arr
# dataset = scaler.fit_transform(mm)
# print(dataset)
# b = a[0:1]
# print(b)
import math
from sklearn.preprocessing import MinMaxScaler
import numpy
def err(true,predicted):
    err = 0
    for i in range(len(true)):
        tmp = (true[i]-predicted[i])/true[i]
        err += tmp*tmp
    standard_err = math.sqrt(err/len(true))
    return standard_err

a = [2,3,3,5]
b = [2,2,3,12]

m = err(b,a)
print(m)