import matplotlib.pylab as plt
import pandas as pd

df = pd.read_csv('user_balance_table_all.csv', index_col='user_id', names=['user_id', 'report_date', 'tBalance', 'yBalance', 'total_purchase_amt', 'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt', 'category1', 'category2', 'category3', 'category4'
], parse_dates=[1])

df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')

labels = ['tBalance', 'yBalance', 'total_purchase_amt', 'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt']
for label in labels:
    print(label)
    df[label] = pd.to_numeric(df[label], errors='coerce')

df = df.groupby('report_date').sum()
ts = df['consume_amt']
print(ts.sort_values())
# df['total_purchase_amt'].plot()
# for label in labels:
#     df[label].sort(ascending=False).plot()
#     plt.title(label)
#     plt.show()




