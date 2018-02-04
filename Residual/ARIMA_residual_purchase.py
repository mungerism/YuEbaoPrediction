from pandas import read_csv
import pandas as pd
import matplotlib.pylab as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import sys
from statsmodels.tsa.arima_model import ARMA
def err(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
def stationarity_test(timeseries):
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

dataframe = read_csv('../file/final_data/arima/ARIMA_purchase.csv',index_col='date', parse_dates=[0])
# dataset = dataframe.values
dataframe['date'] = pd.to_datetime(dataframe['date'], errors='coerce')

sub_total_purchase_residual = dataframe['residual2']

print(sub_total_purchase_residual.describe())

stationarity_test(sub_total_purchase_residual)
sub_total_purchase_residual.describe()
sub_total_purchase_residual.plot()
plt.title('Purchase Residual By Arima')
plt.show()


# 直方图 是否正态分布
sub_total_purchase_residual.hist()
plt.title('Purchase Residual Histogram By Arima')
plt.show()

# autocorrelation
plot_acf(sub_total_purchase_residual, ax=plt.gca(), lags=60)
plt.title('Purchase Residual ACF By Arima')
plt.show()

# LBQ 检验
from statsmodels.stats import diagnostic
print(diagnostic.acorr_ljungbox(sub_total_purchase_residual, lags=None, boxpierce=True))
