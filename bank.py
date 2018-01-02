import matplotlib.pylab as plt
import pandas as pd

df = pd.read_csv('./file/mfd_bank_shibor.csv',
                 parse_dates=[0])

print(df)
df['mfd_date'] = pd.to_datetime(df['mfd_date'], errors='coerce')
labels = ['Interest_O_N','Interest_1_W','Interest_2_W','Interest_1_M','Interest_3_M','Interest_6_M','Interest_9_M','Interest_1_Y']
for label in labels:
    print(label)
    df[label] = pd.to_numeric(df[label], errors='coerce')

df = df.groupby('mfd_date').sum()
df = df[['Interest_O_N', 'Interest_1_W', 'Interest_1_Y']]
df['Interest_O_N'].plot(label='Interest_O_N', linewidth=1)
df['Interest_1_W'].plot(label='Interest_1_W', linewidth=2)
df['Interest_1_Y'].plot(label='Interest_1_Y', linewidth=4)

plt.title('Shanghai Interbank offered rate')
plt.legend()
# Produces correct output when uncommented:
plt.show()
