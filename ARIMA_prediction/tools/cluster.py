# from copy import deepcopy
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# plt.rcParams['figure.figsize'] = (16, 9)
# plt.style.use('ggplot')
#
# # Importing the dataset
# data = pd.read_csv('xclara.csv')
# print("Input Data and Shape")
# print(data.shape)
# data.head()
#
# # Getting the values and plotting it
# f1 = data['V1'].values
# f2 = data['V2'].values
# X = np.array(list(zip(f1, f2)))
# plt.scatter(f1, f2, c='black', s=7)
#
#
#
# '''
# ==========================================================
# scikit-learn
# ==========================================================
# '''
#
# from sklearn.cluster import KMeans
#
# # Number of clusters
# kmeans = KMeans(n_clusters=3)
# # Fitting the input data
# kmeans = kmeans.fit(X)
# # Getting the cluster labels
# labels = kmeans.predict(X)
# # Centroid values
# centroids = kmeans.cluster_centers_
#
# # Comparing with scikit-learn centroids
# print("Centroid values")
# print("Scratch")
# print("sklearn")
# print(centroids) # From sci-kit learn
# plt.show()

import matplotlib.pylab as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas import Series
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
from sklearn.cluster import KMeans
from sklearn import preprocessing


df = pd.read_csv('user_balance_table_all.csv', index_col='user_id', names=['user_id', 'report_date', 'tBalance', 'yBalance', 'total_purchase_amt', 'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt', 'category1', 'category2', 'category3', 'category4'
], parse_dates=[1])

df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
df['total_purchase_amt'] = pd.to_numeric(df['total_purchase_amt'], errors='coerce')
df['total_redeem_amt'] = pd.to_numeric(df['total_redeem_amt'], errors='coerce')
df['purchase_bank_amt'] = pd.to_numeric(df['purchase_bank_amt'], errors='coerce')


df = df.groupby('user_id').sum()
print(df)
df = df.dropna(thresh=2)


import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Getting the values and plotting it
f1 = df['total_purchase_amt']
f2 = df['total_redeem_amt']
X = np.array(list(zip(f1, f2)))
X_scaled = preprocessing.scale(X)
print(X_scaled)


from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X_scaled)
# Getting the cluster labels
labels = kmeans.predict(X_scaled)
# Centroid values
centroids = kmeans.cluster_centers_

colors = ['r','b','y']
makers = ['.', 'o', 'x']

plt.scatter(f1, f2, color=[colors[l_] for l_ in labels], label=labels)

for l in labels:
    plt.scatter(f1, f2, marker=makers[l])

plt.scatter(centroids[:, 0], centroids[:, 1], color=[c for c in colors[:len(centroids)]], marker = "x", s=1, linewidths = 5, zorder = 10)
plt.show()

print(centroids) # From sci-kit learn