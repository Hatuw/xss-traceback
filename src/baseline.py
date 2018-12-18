# coding: utf-8
import os

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
# from keras.layers import Activation
# from keras.layers import Dropout

# set global vars
WORKING_DIR = os.path.split(os.path.realpath(__file__))[0]
DATA_DIR = os.path.join(WORKING_DIR, '../data/data_v1')
DATA_X_FILE = os.path.join(DATA_DIR, 'train.csv')
DATA_Y_FILE = os.path.join(DATA_DIR, 'labels.csv')
HIS_FILE = os.path.join(DATA_DIR, 'history.csv')

# load training data
x_data = np.loadtxt(DATA_X_FILE, dtype=float, delimiter=',')

pre_y_data = np.loadtxt(DATA_Y_FILE, dtype=int)
y_max = pre_y_data.max()
y_data = keras.utils.to_categorical(              # one-hot
    pre_y_data,
    num_classes=y_max+1,
)

# # debug
print("======================================")
# print(len(set(pre_y_data)))
# print(type(pre_y_data))
print(y_max)
print(y_data.shape[0])
# print(y_data.shape[-1])
print("======================================")


# shuffle data and split training set & testing set
data_m = y_data.shape[0]
index = np.arange(data_m)
np.random.shuffle(index)
train_x = x_data[index[:int(data_m*0.8)]]
train_y = y_data[index[:int(data_m*0.8)]]
test_x = x_data[index[int(data_m*0.8):]]
test_y = y_data[index[int(data_m*0.8):]]

# create model
model = Sequential()
model.add(Dense(
    1000,
    activation='relu',
    input_dim=x_data.shape[-1],
))
model.add(Dense(
    2000,
    activation='relu',
    input_dim=1000,
))
# model.add(Dropout(0.5))
model.add(Dense(
    2000,
    activation='relu',
    input_dim=2000,
))
model.add(Dense(
    1000,
    activation='relu',
    input_dim=2000,
))
# model.add(Dropout(0.5))
model.add(Dense(
    y_data.shape[-1],
    activation='softmax',
    input_dim=1000,
))

sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy'],
)

hist = model.fit(
    train_x,
    train_y,
    epochs=20,
    batch_size=128,
)
score = model.evaluate(test_x, test_y, batch_size=128)
print(hist)
with open(HIS_FILE, 'a', encoding='UTF-8', newline="") as f:
    f.write(str(hist.history))
