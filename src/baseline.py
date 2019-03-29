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
from sklearn.model_selection import  KFold

# from keras.optimizers import SGD
# from keras.models import load_model

# from keras.layers import Activation
# from keras.layers import Dropout

# set global vars
# WORKING_DIR = os.path.split(os.path.realpath(__file__))[0]
# DATA_DIR = os.path.join(WORKING_DIR, '../data/data_v1')
# Result_dir = "../result"
# DATA_X_FILE = os.path.join(DATA_DIR, 'train.csv')
# DATA_Y_FILE = os.path.join(DATA_DIR, '')
# HIS_FILE = os.path.join(Result_dir, 'history.csv')

import functools
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top10_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=10)

top10_acc.__name__ = 'top10_acc'
top3_acc.__name__ = 'top3_acc'
class BaselineConfig(Config):
    """Configuration for the baseline training."""
    NAME = "baseline"

config = BaselineConfig()
print(config.DATA_DIR)
# exit()

# load training data
x_data = np.loadtxt(Config.DATA_X_FILE, dtype=float, delimiter=',')
pre_y_data = np.loadtxt(Config.DATA_Y_FILE, dtype=int)
print(len(x_data))
print(len(pre_y_data))
y_max = pre_y_data.max()
y_data = keras.utils.to_categorical(              # one-hot
    pre_y_data,
    num_classes=y_max+1,
)
'''
# shuffle data and split training set & testing set
data_m = y_data.shape[0]
index = np.arange(data_m)
np.random.shuffle(index)
train_x = x_data[index[:int(data_m*0.8)]]
train_y = y_data[index[:int(data_m*0.8)]]
test_x = x_data[index[int(data_m*0.8):]]
test_y = y_data[index[int(data_m*0.8):]]
# print(x_data.shape[-1])
'''
# print(train_test_split(x_data, y_data))
kf = KFold(n_splits=3, shuffle=True)
for train, test in kf.split(x_data):
    start = time.clock()
    print("TRAIN:", train, "TEST:", test)
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

    print(y_data.shape[-1])

    # sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop,
        metrics=['accuracy', top3_acc, 'top_k_categorical_accuracy', top10_acc],
    )
    # Fit the model
    hist = model.fit(
        x_data[train],
        y_data[train],
        epochs=20,
        batch_size=128,
    )

    # evaluate the model
    loss, accuracy, accuracy_top3, accuracy_top5, accuracy_top10 = model.evaluate(x_data[test], y_data[test])  # , batch_size=128)

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)
    print('\ntest accuracy_top3: ', accuracy_top3)
    print('\ntest accuracy_top5: ', accuracy_top5)
    print('\ntest accuracy_top10: ', accuracy_top10)

    # save result
    with open(Config.HIS_FILE, 'a', encoding='UTF-8', newline="") as f:
        f.write(str(hist.history) + "\n")
        test_str = "test result: "
        test_str += "loss : " + str(loss) + ";"
        test_str += "accuracy : " + str(accuracy) + ";"
        test_str += "accuracy_top3 : " + str(accuracy_top3) + ";"
        test_str += "accuracy_top5 : " + str(accuracy_top5) + ";"
        test_str += "accuracy_top10 : " + str(accuracy_top10) + ";"
        f.write(test_str + "\n")
        f.write("\n")
        f.close()

    print(hist.history.keys())
    # plot acc
    fig1 = plt.figure()
    plt.plot(hist.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.show()
    now_name = time.strftime("%Y-%m-%d%H%M%S", time.localtime()) + "_acc.png"
    Result_png = os.path.join(Config.RESULT_DIR, now_name)
    fig1.savefig(Result_png)
    # plot loss
    fig2 = plt.figure()
    plt.plot(hist.history['loss'])
    plt.title('model loss')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.show()
    now_name = time.strftime("%Y-%m-%d%H%M%S", time.localtime()) + "_loss.png"
    Result_png = os.path.join(Config.RESULT_DIR, now_name)
    fig2.savefig(Result_png)
    # plt.tight_layout()
    # plt.plot(hist.history['loss'])
    # plt.legend(['acc', 'loss'], loc='upper left')
    # plt.tight_layout()
    # save result png

    # save
    Result_mode = os.path.join(Config.RESULT_DIR, now_name) + "_model.h5"

    # HDF5 file, you have to pip3 install h5py if don't have it
    model.save(Result_mode)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
