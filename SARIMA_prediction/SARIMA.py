import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../file/user_balance_table_all.csv', index_col='user_id', names=['user_id', 'report_date', 'tBalance', 'yBalance', 'total_purchase_amt', 'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt', 'category1', 'category2', 'category3', 'category4'
], parse_dates=[1])

df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
df['total_purchase_amt'] = pd.to_numeric(df['total_purchase_amt'], errors='coerce')
df['total_redeem_amt'] = pd.to_numeric(df['total_redeem_amt'], errors='coerce')
df['purchase_bank_amt'] = pd.to_numeric(df['purchase_bank_amt'], errors='coerce')

df = df.groupby('report_date').sum()
ts = df['total_purchase_amt']
ts = ts['2014-04-01':'2014-06-30']

model=sm.tsa.statespace.SARIMAX(endog=ts,order=(1,1,0),seasonal_order=(0,0,0,7),trend='c',enforce_invertibility=False)
results=model.fit()
print(results.summary())
print(results.predict())

predict_ts = results.predict()

predict_ts.plot(label='predicted')
ts.plot(label='original')
plt.legend(loc='best')
plt.show()




