from pandas import read_csv
import matplotlib.pylab as plt
import numpy as np
dataframe = read_csv("./0218_mix_data_pur.csv")
dataset = dataframe.values
error = dataset[:,2]
error = np.array((error))
arr = np.linspace(0,1,100)
# plt.xlim((0,1))
# plt.xaxixs(arr)
plt.xlabel("YEB_LSTM portion(%)")
plt.title("Purchase varition trend")
plt.plot(error)
plt.show()