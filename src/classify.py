# coding: utf-8
from time import time
import logging
from sklearn import svm
import fix_data
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import tensorflow as tf
import numpy as np
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def load_fixed_data(maxlen = 9999):
    # read fixed_data
    f = open('../data/fixed_data.csv', 'r',encoding='UTF-8')
    result = {}
    reader = csv.reader(f)
    for item in enumerate(reader):
        if(item[0]>=maxlen):
            break

        tmp = item[1]
        if(len(tmp) == 0):
            print("something wrong......")
        result[item[0]] = item[1]
    f.close()
    return  result


def load_data():
    X = []
    X_test = []
    Y_test = []
    Y = []
    fixed_data_train = fix_data.load_fixed_data()
    fixed_data_test = fix_data.load_fixed_data()
    fixed_data_train = list(fixed_data_train.values())
    fixed_data_test = list(fixed_data_test.values())
    for i in range(len(fixed_data_train)-1000):
        fixed_data_train_X = [float(x) for x in fixed_data_train[i][:2500]]
        fixed_data_train_Y = [float(x) for x in fixed_data_train[i][2500:]]
        X.append(fixed_data_train_X)
        Y.append(fixed_data_train_Y)
    # print(len(fixed_data_train[0][2500:]))
    # print(type(fixed_data_train[0][:2500][1]))
    # print(type(fixed_data_train[0]))
    for i in range(5000,6000):
        fixed_data_test_X = [float(x) for x in fixed_data_test[i][:2500]]
        fixed_data_test_Y = [float(x) for x in fixed_data_train[i][2500:]]
        X_test.append(fixed_data_test_X)
        Y_test.append(fixed_data_test_Y)
    # print(len(X))
    # print(len(Y))
    return X,Y,X_test,Y_test


def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs,Weights,biases


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # print(v_ys[0])
    #     # print(type(v_ys[0]))
    #     # print(y_pre)
    #     # print(type(y_pre[0]))
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

def next_batch(train_data, train_target, batch_size):
    index = [ i for i in range(0,len(train_target)) ]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0,batch_size):   #batch_size-->number of every time you get the length of data
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target


print("begin classify......")
begin = time()
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 2500])  # 50*50
ys = tf.placeholder(tf.float32, [None, 1050])

# add output layer
prediction,Weights,biases = add_layer(xs, 2500, 1050, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
init = tf.global_variables_initializer()
sess.run(init)
X,Y,X_test,Y_test = load_data()
for i in range(1000):
    batch_xs, batch_ys = next_batch(X,Y,4000)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    # print(sess.run(Weights, feed_dict={xs: batch_xs, ys: batch_ys}))
    # print(sess.run(biases, feed_dict={xs: batch_xs, ys: batch_ys}))
    test_xs, test_ys = next_batch(X,Y,4000)
    if i % 50 == 0:
        print(compute_accuracy(X_test, Y_test))


end = time()
print("Total procesing time: %d seconds" % (end - begin))




#
# if __name__ == '__main__':
#     main()