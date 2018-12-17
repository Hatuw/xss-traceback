# coding: utf-8
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# set global vars
WORKING_DIR = os.path.split(os.path.realpath(__file__))[0]
DATA_DIR = os.path.join(WORKING_DIR, '../data')
DATA_X_FILE = os.path.join(DATA_DIR, 'train.csv')
DATA_Y_FILE = os.path.join(DATA_DIR, 'labels.csv')

# load training data
x_data = np.loadtxt(DATA_X_FILE, dtype=float, delimiter=',')

pre_y_data = np.loadtxt(DATA_Y_FILE, dtype=int)
y_max = pre_y_data.max()
y_data = keras.utils.to_categorical(
    pre_y_data,
    num_classes=y_max+1
)

# create model
model = Sequential()
model.add(Dense(
    2000,
    activation='relu',
    input_dim=x_data.shape[-1]
))
# model.add(Dropout(0.5))
model.add(Dense(
    2000, activation='relu'
))
# model.add(Dropout(0.5))
model.add(Dense(
    y_data.shape[-1],
    activation='softmax'
))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(
    x_data,
    y_data,
    epochs=20,
    batch_size=128
)
score = model.evaluate(x_data, y_data, batch_size=128)
