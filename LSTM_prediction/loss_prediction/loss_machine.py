'''
赎回的lstm
http://blog.csdn.net/youyuyixiu/article/details/72841703
http://blog.csdn.net/youyuyixiu/article/details/72840893
'''
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras.models
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential

import pandas as pd
from sklearn.externals import joblib
import os

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
def err(true,predicted):
    err = 0
    for i in range(len(true)):
        tmp = (true[i]-predicted[i])/true[i]
        err += tmp*tmp
    standard_err = math.sqrt(err/len(true))
    return standard_err

def create_dataset(dataset_X, dataset_Y, look_back=1):
    dataX, dataY = [], []
    dataX = dataset_X[0:len(dataset_Y)-look_back-1]
    for i in range(len(dataset_Y)-look_back-1):
        dataY.append(dataset_Y[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# # load the dataset
# dataframe = read_csv('./file/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataset = dataset.astype('float32')
# plt.plot(dataset)
# plt.show()

dataframe = read_csv('../../file/hybrid_total_redeem.csv', usecols=[7], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float64')
plt.plot(dataset)
plt.show()

dataframe_mulfeature = read_csv('../../file/hybrid_total_redeem.csv', usecols=[1,2,3,4,5,6,7,8,9,10,11,12], engine='python', skipfooter=3)
dataset_mulfeature = dataframe_mulfeature.values
dataset_mulfeature = dataset_mulfeature.astype('float64')


# fix random seed for reproducibility
numpy.random.seed(7)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
scaler2 = MinMaxScaler(feature_range=(0, 1))
dataset_mulfeature = scaler2.fit_transform(dataset_mulfeature)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train_x, test_x = dataset_mulfeature[0:train_size,:], dataset_mulfeature[train_size:len(dataset),:]
train_y,test_y = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# use this function to prepare the train and test datasets for modeling
look_back = 1
trainX, trainY = create_dataset(train_x,train_y, look_back)
testX, testY = create_dataset(test_x,test_y, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print("trainX",trainX.shape)
print("trainY",trainY.shape)
print("testX",testX.shape)
print("testY",testY.shape)

clf = RandomForestRegressor(n_estimators=200,max_features = .12).fit(trainX,trainY)
# create and fit the LSTM network
trainPredict = clf.predict(trainX)
testPredict = clf.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

#     joblib.dump(model_prob, "lstm.model")
# clf = joblib.load("lstm.model")
# make predictions
# trainPredict = clf.predict(trainX)
# testPredict = clf.predict(testX)

# invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
errs = err(testY[0],testPredict[:,0])
print("errs:",errs)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

