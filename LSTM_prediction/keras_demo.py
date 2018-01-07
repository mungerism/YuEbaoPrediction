from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,LSTM,Reshape
from sklearn import preprocessing
import os
import numpy as np
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras import backend as K
import pprint, pickle
def load_data():
    inDir = "E:\morindaz\STFT\small5\\"
    pkl_file_xtrain = open(inDir + 'Train5_x.pkl', 'rb')
    pkl_file_ytrain = open(inDir + 'Train5_y.pkl', 'rb')
    pkl_file_xtest = open(inDir +'Test5_x.pkl', 'rb')
    pkl_file_ytest = open(inDir +'Test5_y.pkl', 'rb')
    X_train = pickle.load(pkl_file_xtrain)
    X_test = pickle.load(pkl_file_xtest)
    y_train = pickle.load(pkl_file_ytrain)
    y_test = pickle.load(pkl_file_ytest)
    pkl_file_xtest.close()
    pkl_file_ytest.close()
    pkl_file_xtrain.close()
    pkl_file_ytrain.close()
    return X_train,X_test,y_train,y_test
    # print(data1)
# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
batch_size = 128
# 0-9手写数字一个有10个类别
num_classes = 4
# 12次完整迭代，差不多够了
epochs = 2
# 输入的图片是28*28像素的灰度图
# img_rows, img_cols = 28, 28
# 训练集，测试集收集非常方便
x_train,x_test,y_train,y_test = load_data()
x_train2 = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=350, dtype='int32',
    padding='post', truncating='post', value=0.)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=350, dtype='int32',
    padding='post', truncating='post', value=0.)

print(x_train2.shape)
print(x_test.shape)

mms = preprocessing.StandardScaler()
result_train = []
result_test = []
for i in range(len(x_train2)):
    tmp_train = mms.fit_transform(x_train2[i])
    result_train.append(tmp_train)
# print(result_train.shape)
for j in range(len(x_test)):
    tmp_test = mms.fit_transform(x_test[j])
    result_test.append(tmp_test)

result_train = np.array(result_train)
result_test = np.array(result_test)

print(result_train.shape, result_test.shape)

X_train= result_train.reshape((result_train.shape[0],350,513,1))
x_test = result_test.reshape((x_test.shape[0],350,513, 1))
print(X_train.shape)
print(x_test.shape)
# keras.layers.normalization.BatchNormalization(epsilon=1e-6, weights=None)
# X_test = preprocessing.scale(x_test)
# X_train = preprocessing.scale(x_train2)
# 把类别0-9变成2进制，方便训练
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
# 牛逼的Sequential类可以让我们灵活地插入不同的神经网络层
# model = Sequential()
# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
# 卷积核的窗口选用3*3像素窗口
nb_classes = 4
model = Sequential()

model.add(Conv2D(32,3, 3, border_mode='same', input_shape = (350,513,1)))
print(model.output_shape)
model.add(Conv2D(32,3, 3, border_mode='same'))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))
model.add(Conv2D(32,3, 3, border_mode='same'))
print(model.output_shape)
model.add(Conv2D(32,3, 3, border_mode='same'))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))
model.add(Conv2D(32,3, 3, border_mode='same'))
print(model.output_shape)
model.add(Conv2D(32,3, 3, border_mode='same'))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))
model.add(Conv2D(32,3, 3, border_mode='same'))
print(model.output_shape)
model.add(Conv2D(32,3, 3, border_mode='same'))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))

# model.add(Convolution2D(32, 5, 5))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(poolsize=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(64, 48, 3, 3, border_mode='full'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(poolsize=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(80, 64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(poolsize=(2, 2)))
# model.add(Dropout(0.25))
#
model.add(Flatten())
print(model.output_shape)
# model.add(Dense(80*8*10, 1000))
# model.add(Activation('relu'))
# model.add(BatchNormalization((1000,)))
# model.add(Dropout(0.5))
#
# model.add(Dense(1000, 1000))
# model.add(Activation('relu'))
# model.add(BatchNormalization((1000,)))
# model.add(Dropout(0.5))
model.add(Reshape((21, -1)))
print(model.output_shape)
#ss
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


opt = Adadelta(lr=0.00001)

fname = 'gender_best_weights.hdf5'
if os.path.isfile(fname):
    model.load_weights(fname)
    # print "load weights successful!"
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
print(model.summary())
checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
early_stop = EarlyStopping(patience=20, verbose=1)
model.fit(X_train, y_train, batch_size=8, epochs=100, verbose=1, validation_data=(x_test, y_test))

# print "Test score: {0}".format(score)
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])