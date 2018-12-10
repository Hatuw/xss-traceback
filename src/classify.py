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



def load_fixed_data(maxlen = 99999):
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

def np2list(array):
    list = array.tolist()
    return list

def list2np(x):
    array = np.array(x)
    return array

def normalization(x):
    x = list2np(x)
    return np2list(((x - min(x)) / (max(x) - min(x))))

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
        norm_train_x = normalization(fixed_data_train_X)
        norm_train_y = normalization(fixed_data_train_Y)

        X.append(norm_train_x)
        Y.append(norm_train_y)
    # print(len(fixed_data_train[0][2500:]))
    # print(type(fixed_data_train[0][:2500][1]))
    # print(type(fixed_data_train[0]))
    for i in range(5000,6000):
        fixed_data_test_X = [float(x) for x in fixed_data_test[i][:2500]]
        fixed_data_test_Y = [float(x) for x in fixed_data_train[i][2500:]]
        # print(fixed_data_test_Y)
        norm_test_x = normalization(fixed_data_test_X)
        norm_test_y = normalization(fixed_data_test_Y)
        X_test.append(norm_test_x)
        Y_test.append(norm_test_y)
    print(len(X))
    print(len(Y))
    return X,Y,X_test,Y_test

# def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
#     # add one more layer and return the output of this layer
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#     biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases
#     if activation_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_function(Wx_plus_b,)
#     return outputs, Weights, biases, Wx_plus_b


def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
    # weights and biases (bad initialization for this case)
    Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # fully connected product
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # normalize fully connected product
    if norm:
        # Batch Normalize
        # axes 这批数据的哪几个维度上求均值与方差
        # the dimension you wanna normalize, axes[0] for batch只要0维度上就OK
        # for image 三个维度上求均值方差, you wanna do [0, 1, 2] for [batch, height, width] but not channel
        fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0, 1, 2])


        # # 计算Wx_plus_b 的均值与方差,其中axis = [0] 表示想要标准化的维度
        # img_shape = [128, 32, 32, 64]
        # Wx_plus_b = tf.Variable(tf.random_normal(img_shape))
        # axis = list(range(len(img_shape) - 1))  # [0,1,2]
        # wb_mean, wb_var = tf.nn.moments(Wx_plus_b, axis)


        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001
        # 应用均值和变量的移动平均值,BN在神经网络进行training和testing的时候，所用的mean、variance是不一样的
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()

        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
        # similar with this two steps:
        # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
        # scale为扩大的参数,shift为平移的参数
        # Wx_plus_b = Wx_plus_b * scale + shift

    # activation
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs,Weights,biases


# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs})
#     # print(y_pre)
#     # print(v_ys[0])
#     #     # print(type(v_ys[0]))
#     #     # print(y_pre)
#     #     # print(type(y_pre[0]))
#     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#     return result
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    # print("correct_prediction = ")
    # print(sess.run(correct_prediction, feed_dict={xs: v_xs, ys: v_ys}))
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
ys = tf.placeholder(tf.float32, [None, 50])

# add output layer
prediction,Weights,biases = add_layer(xs, 2500, 50, activation_function=tf.nn.softmax, norm=False)#, activation_function=tf.nn.softmax)

# # the error between prediction and real data
# cross_entropy = tf.reduce_mean(tf.reduce_sum(ys - prediction,
#                                               reduction_indices=[1]))  # loss
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
cross_entropy = -tf.reduce_sum(ys * tf.log(prediction * 1e-10))
# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
init = tf.global_variables_initializer()
sess.run(init)
X,Y,X_test,Y_test = load_data()
print(np.var(X[1]))
print(len(X))
for i in range(2000):
    batch_xs, batch_ys = next_batch(X,Y,8000)
    # print(len(batch_xs[1]))
    # print(len(batch_ys[1]))
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    # print(sess.run(Weights, feed_dict={xs: batch_xs, ys: batch_ys}))
    # print(sess.run(biases, feed_dict={xs: batch_xs, ys: batch_ys}))
    # print(sess.run(X[1], feed_dict={xs: batch_xs, ys: batch_ys}))
    # print(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}))

    test_xs, test_ys = next_batch(X,Y,5000)
    if i % 50 == 0:
        print("round "+ str(i))
        print(sess.run(prediction, feed_dict={xs: batch_xs, ys: batch_ys}))
        print(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}))
        print(compute_accuracy(X_test, Y_test))


end = time()
print("Total procesing time: %d seconds" % (end - begin))




#
# if __name__ == '__main__':
#     main()