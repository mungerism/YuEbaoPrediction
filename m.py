import math
def err(true,predicted):
    err = 0
    for i in range(len(true)):
        tmp = (true[i]-predicted[i])/true[i]
        err += math.sqrt(tmp*tmp)
    standard_err = err/len(true)
    return standard_err
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
a = [2,3,4,5]
a = np.array(a)
b = [2,3,3,6]
b = np.array(b)
errs = err(a,b)
print(errs)

err2 = mean_absolute_percentage_error(a,b)
print(err2)