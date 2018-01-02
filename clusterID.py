import matplotlib.pylab as plt
import pandas as pd
from sklearn.manifold import Isomap,MDS,TSNE
from sklearn.cluster import KMeans
from sklearn import preprocessing
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('./file/user_balance_table.csv', index_col='user_id', names=['user_id', 'report_date', 'tBalance', 'yBalance', 'total_purchase_amt', 'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt', 'category1', 'category2', 'category3', 'category4'
], parse_dates=[1])

df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
df['total_purchase_amt'] = pd.to_numeric(df['total_purchase_amt'], errors='coerce')
df['total_redeem_amt'] = pd.to_numeric(df['total_redeem_amt'], errors='coerce')
df['direct_purchase_amt'] = pd.to_numeric(df['direct_purchase_amt'], errors='coerce')
df['purchase_bal_amt'] = pd.to_numeric(df['purchase_bal_amt'], errors='coerce')
df['purchase_bank_amt'] = pd.to_numeric(df['purchase_bank_amt'], errors='coerce')
df['consume_amt'] = pd.to_numeric(df['consume_amt'], errors='coerce')
df['transfer_amt'] = pd.to_numeric(df['transfer_amt'], errors='coerce')

df = df.groupby('user_id').sum()
print(df)
df = df.dropna(thresh=2)

plt.rcParams['figure.figsize'] = (100, 100)
plt.style.use('ggplot')

# Getting the values and plotting it
f1 = df['total_purchase_amt']
f2 = df['total_redeem_amt']
f3 = df['direct_purchase_amt']
f4 = df['purchase_bal_amt']
f5 = df['purchase_bank_amt']
f6 = df['consume_amt']
f7 = df['transfer_amt']
X = np.array(list(zip(f1, f2)))

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y = kmeans.labels_
centers = kmeans.cluster_centers_
marker = ['+','1','o']
color = ['r','y','k']
color_ = ['red point','yellow point','black point','blue point','green point']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('frame clustering')
# a[1]
for j in range(3):
   z = np.argwhere(y==j)
   ax.scatter(X[z,0],X[z,1],c = color[j],marker = marker[j])
plt.show()
print (111)