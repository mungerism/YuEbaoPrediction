'''
申购的lstm取得较为不错的效果
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
from keras.models import Sequential

import pandas as pd
from sklearn.externals import joblib
import os

# Xsss is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
def create_dataset(data_x,dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = data_x[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# # load the dataset
# dataframe = read_csv('./file/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataset = dataset.astype('float32')
# plt.plot(dataset)
# plt.show()

dataframe_amt = read_csv('../file/grouped.csv', usecols=[4], engine='python', skipfooter=3)
dataset = dataframe_amt.values
print(dataset)
dataset = dataset.astype('float64')
plt.plot(dataset)
plt.show()
print(dataset)
dataframe = read_csv('../file/grouped.csv', usecols=[1,2,3,4,5,6,7,8,9,10,11,12],  engine='python', skipfooter=3)
dataframe_value = dataframe.values
dataframe_type = dataframe_value.astype("float64")
# fix random seed for reproducibility
numpy.random.seed(7)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
dataset_second = scaler.fit_transform(dataframe_type)

# split into train and test sets
train_size = int(len(dataset_second) * 0.67)
test_size = len(dataset_second) - train_size
train_Y, test_Y = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#includes 12 features
train_X, test_X = dataset_second[0:train_size,:], dataset_second[train_size:len(dataset),:]

# use this function to prepare the train and test datasets for modeling
look_back = 1
trainX, trainY = create_dataset(train_X,train_Y, look_back)
testX, testY = create_dataset(test_X,test_Y, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

model_prob = model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

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
print("hh")

