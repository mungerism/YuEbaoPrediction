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
import pandas as pd
import numpy as np
import sys

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=30)
    rolstd = pd.rolling_std(timeseries, window=30)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def _proper_model(ts_log_diff, maxLag):
    best_p = 0
    best_q = 0
    best_bic = sys.maxsize
    best_model=None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(ts_log_diff, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARMA.bic
            print (bic, best_bic)
            if bic < best_bic:
                best_p = p
                best_q = q
                best_bic = bic
                best_model = results_ARMA
    return best_p,best_q,best_model

df = pd.read_csv('user_balance_table_all.csv', index_col='user_id', names=['user_id', 'report_date', 'tBalance', 'yBalance', 'total_purchase_amt', 'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt', 'category1', 'category2', 'category3', 'category4'
], parse_dates=[1])

df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
df['total_purchase_amt'] = pd.to_numeric(df['total_purchase_amt'], errors='coerce')
df['total_redeem_amt'] = pd.to_numeric(df['total_redeem_amt'], errors='coerce')
df['purchase_bank_amt'] = pd.to_numeric(df['purchase_bank_amt'], errors='coerce')

df = df.groupby('report_date').sum()
ts = df['total_purchase_amt']
ts = ts['2013-07-01':'2014-01-01']

ts.plot()
plt.title('Total purchase')
plt.show()

test_stationarity(ts)


differenced = ts.diff(1)
differenced = differenced[1:]
test_stationarity(differenced)
differenced.plot()
plt.title('Total purchase First difference')
plt.show()



# differenced = differenced.diff(1)
# differenced = differenced[1:]
# test_stationarity(differenced)
# differenced.plot()
# plt.title('Total purchase Second difference')
# plt.show()

plt.figure()
plt.axhline(y=-1.96/np.sqrt(len(differenced)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(differenced)),linestyle='--',color='gray')
plot_acf(differenced, ax=plt.gca(), lags=20)
plt.show()

plt.axhline(y=-1.96/np.sqrt(len(differenced)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(differenced)),linestyle='--',color='gray')
plot_pacf(differenced, ax=plt.gca(), lags=20)
plt.show()

# print(_proper_model(differenced, 9))
#
# model = ARIMA(ts, order=(1, 1, 0))
# results_AR = model.fit(disp=-1)
# plt.plot(differenced)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-differenced)**2))
# plt.show()

model = ARIMA(ts, order=(0, 1, 2))
results_MA = model.fit(disp=-1)
plt.plot(differenced, label = 'Original')
plt.plot(results_MA.fittedvalues, color='red', label = 'Fitted values')
plt.title('ARIMA')
plt.legend(loc='best')
# plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-differenced)**2))
plt.show()


# model = ARIMA(ts, order=(8, 1, 4))
# results_ARIMA = model.fit(disp=-1)
# plt.plot(differenced)
# plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-differenced)**2))
# plt.show()

predictions_ARIMA_diff = pd.Series(results_MA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(differenced.ix[0], index=differenced.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()

plt.plot(ts, label='Original')
plt.plot(0.5*0.5*predictions_ARIMA_log, label='Prediction')
plt.legend(loc='best')

# plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA_log - ts)**2)/len(ts)))
plt.show()





