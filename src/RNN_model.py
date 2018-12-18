# coding: utf-8
import os
import keras
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
# from keras.optimizers import Adam

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
y_data = keras.utils.to_categorical(  # one-hot
    pre_y_data,
    num_classes=y_max + 1,
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
train_x = x_data[index[:int(data_m * 0.8)]]
train_y = y_data[index[:int(data_m * 0.8)]]
test_x = x_data[index[int(data_m * 0.8):]]
test_y = y_data[index[int(data_m * 0.8):]]
print(x_data.shape)
'''
# create model
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',  # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',  # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
'''