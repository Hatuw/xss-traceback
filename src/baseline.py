# coding: utf-8
import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
# from keras.optimizers import SGD
# from keras.models import load_model

# from keras.layers import Activation
# from keras.layers import Dropout

# set global vars
# WORKING_DIR = os.path.split(os.path.realpath(__file__))[0]
# DATA_DIR = os.path.join(WORKING_DIR, '../data/data_v1')
# Result_dir = "../result"
# DATA_X_FILE = os.path.join(DATA_DIR, 'train.csv')
# DATA_Y_FILE = os.path.join(DATA_DIR, 'labels.csv')
# HIS_FILE = os.path.join(Result_dir, 'history.csv')


class BaselineConfig(Config):
    """Configuration for the baseline training."""
    NAME = "baseline"


config = BaselineConfig()
print(config.DATA_DIR)
exit()

# load training data
x_data = np.loadtxt(Config.DATA_X_FILE, dtype=float, delimiter=',')

pre_y_data = np.loadtxt(Config.DATA_Y_FILE, dtype=int)
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

# sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(
    loss='categorical_crossentropy',
    optimizer=rmsprop,
    metrics=['accuracy'],
)

hist = model.fit(
    train_x,
    train_y,
    epochs=20,
    batch_size=128,
)
loss, accuracy = model.evaluate(test_x, test_y, batch_size=128)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
with open(Config.HIS_FILE, 'a', encoding='UTF-8', newline="") as f:
    f.write(str(hist.history) + "\n")
    f.close()

print(hist.history.keys())

fig = plt.figure()
plt.plot(hist.history['acc'])
plt.title('model accuracy-loss')
plt.ylabel('value')
plt.xlabel('epoch')
plt.tight_layout()
plt.plot(hist.history['loss'])
plt.legend(['acc', 'loss'], loc='upper left')
plt.tight_layout()
plt.show()
# save result png
now_name = time.strftime("%Y-%m-%d%H%M%S", time.localtime()) + "_acc-loss.png"
Result_png = os.path.join(Config.RESULT_DIR, now_name)
fig.savefig(Result_png)

# save
Result_mode = os.path.join(Config.RESULT_DIR, now_name) + "_model.h5"

# HDF5 file, you have to pip3 install h5py if don't have it
model.save(Result_mode)
# del model  # deletes the existing model

# # load
# model = load_model(Result_mode)
