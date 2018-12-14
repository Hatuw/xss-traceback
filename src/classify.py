# coding: utf-8
import csv
import numpy as np
import tensorflow as tf
import time
import logging
import warnings
import os
import pandas as pd
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# set global vars
DATA_DIR = '../data'
# mirror_urls_file = 'mirror_demo.csv'  # toy dataset
mirror_urls_file = 'train.csv'  # train full dataset


def np2list(array):
    list = array.tolist()
    return list


def list2np(x):
    array = np.array(x)
    return array


def normalization(x):
    x = list2np(x)
    return np2list(((x - min(x)) / (max(x) - min(x))))

def myone_hot(labels):
    labels = set(labels)
    print("labels len = " + str(len(labels)))
    labels_key = {}
    for i, value in enumerate(labels):
        labels_key[value] = i
    return labels_key

def load_data():
    #load train_url data
    global mirror_urls_file
    file_path = os.path.join(DATA_DIR, mirror_urls_file)
    assert os.path.exists(file_path), "file \"{}\" not exist".format(file_path)
    data = pd.read_csv(file_path,header=None)
    train_url_data = data.values
    # normalization data
    for i,url in enumerate(train_url_data):
        train_url_data[i] = normalization(url)
    # print(train_url_data[0])

    # load train_labels_data of labels
    mirror_urls_file = 'labels.csv'
    file_path = os.path.join(DATA_DIR, mirror_urls_file)
    assert os.path.exists(file_path), "file \"{}\" not exist".format(file_path)
    with open(file_path, 'r', encoding='UTF-8') as f:
        train_author_data = []
        reader = csv.reader(f)
        for item in enumerate(reader):
            tmp = item[1]
            train_author_data.append(int(tmp[0]))
        # print(type(train_author_data[0]))
        # print(len(train_author_data))

    # load train_labels_data of labels_map
    mirror_urls_file = 'labels_map.csv'
    file_path = os.path.join(DATA_DIR, mirror_urls_file)
    assert os.path.exists(file_path), "file \"{}\" not exist".format(file_path)
    with open(file_path, 'r', encoding='UTF-8') as f:
        train_author_map_data = []
        reader = csv.reader(f)
        for item in enumerate(reader):
            tmp = item[1]
            train_author_map_data.append(tmp[0])
        # print(train_author_map_data[1])
        # print(len(train_author_map_data))
    return train_url_data,train_author_data,train_author_map_data

def save_data(time,learning_rate,training_epochs,accuracy):
    global mirror_urls_file
    mirror_urls_file = 'experimental_data.csv'
    file_path = os.path.join(DATA_DIR, mirror_urls_file)
    with open(file_path, 'a', encoding='UTF-8',newline="") as f:
        writer = csv.writer(f)
        data = np.array([time,learning_rate,training_epochs,accuracy])
        # f.write(data)
        writer.writerow(data)
        f.close()


def add_layer(inputs, in_size, out_size,
              activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs, Weights, biases,Wx_plus_b,outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # print("correct_prediction = ")
    # print(sess.run(correct_prediction, feed_dict={xs: v_xs, ys: v_ys}))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    y_pre = np2list(y_pre)
    y_pre_max_index = sess.run(tf.argmax(y_pre, 1), feed_dict={xs: v_xs})
    v_ys_max_index = sess.run(tf.argmax(v_ys, 1), feed_dict={ys: v_ys})
    # correct_prediction = np2list(correct_prediction)
    return result,y_pre_max_index,v_ys_max_index,y_pre.index(max(y_pre)),y_pre


def next_batch(train_data, train_target, batch_size):
    index = [i for i in range(0, len(train_target))]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    # batch_size-->number of every time you get the length of data
    for i in range(current_batch*batch_size, current_batch*batch_size + batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target

def myone_hot_author(batch_ys):
    for i, author_index in enumerate(batch_ys):
        new_batch_ys_tmp = np.zeros(2626)
        new_batch_ys_tmp[author_index] = 1
        new_batch_ys_tmp = new_batch_ys_tmp.tolist()
        batch_ys[i] = new_batch_ys_tmp
    return batch_ys

def main():
    global sess
    global xs
    global ys
    global prediction
    #learning rate
    learning_rate = 0.01
    print("begin classify......")
    begin = time.time()
    train_url_data, train_author_data, train_author_map_data = load_data()
    # print(train_author_data)
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 100])  # 50*50
    ys = tf.placeholder(tf.float32, [None, 2626])

    # add output layer
    fcn1, _, _,_,_ = add_layer(xs, 100, 1000, activation_function=tf.nn.softmax)
    fcn2, _, _,_,_ = add_layer(fcn1, 1000, 2000, activation_function=tf.nn.softmax)
    fcn3, _, _,_,_ = add_layer(fcn2, 2000, 2500, activation_function=tf.nn.softmax)
    prediction, Weights, biases , Wx_plus_b ,outputs= add_layer(
        fcn3, 2500, 2626, activation_function=tf.nn.softmax)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


    sess = tf.Session()
    # important step
    # tf.initialize_all_variables() no long valid from
    init = tf.global_variables_initializer()
    sess.run(init)

    training_epochs = 1000
    batch_size = 5000
    for j in range(training_epochs):
        total_batch = int(len(train_url_data) / batch_size)
        global current_batch
        for current_batch in range(total_batch):
            batch_xs, batch_ys = next_batch(train_url_data, train_author_data, batch_size)
            # print(len(batch_xs[1]))
            # print(len(batch_ys[1]))
            # print(batch_ys)

            #one-hot anthor_data before train
            batch_ys = myone_hot_author(batch_ys)
            # print(len(batch_ys))

            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

            # debug code
            # print("===================================================")
            # # print(sess.run(Weights, feed_dict={xs: batch_xs, ys: batch_ys}))
            # # print(sess.run(biases, feed_dict={xs: batch_xs, ys: batch_ys}))
            # print(sess.run(Wx_plus_b, feed_dict={xs: batch_xs, ys: batch_ys}))
            # print(sess.run(outputs, feed_dict={xs: batch_xs, ys: batch_ys}))
            # print(len(sess.run(Wx_plus_b, feed_dict={xs: batch_xs, ys: batch_ys})))
            # print(len(sess.run(outputs, feed_dict={xs: batch_xs, ys: batch_ys})))
            # print("===================================================")
            # print(sess.run(X[1], feed_dict={xs: batch_xs, ys: batch_ys}))
            # print(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}))

        test_xs, test_ys = next_batch(train_url_data, train_author_data, batch_size)
        test_ys = myone_hot_author(test_ys)
        if j % 1 == 0:
            print("round " + str(j))
            # print(len(sess.run(prediction[1], feed_dict={xs: batch_xs, ys: batch_ys})))
            cross_entropy_value = sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys})
            print('Loss at step %d: %f' % (j, cross_entropy_value))
            accuracy,prediction_index1,prediction_index2,y_pre_max,y_pre = compute_accuracy(test_xs, test_ys)
            print('Validation accuracy: %.3f%%' % (accuracy*100))
            # arr = np.arange(2626)
            # for i in range(batch_size):
            # print(len(prediction_index1))
            print(prediction_index1)
            print(prediction_index2)
            # print(y_pre_max)
            # print(len(y_pre))
            # save experimental data
            save_data(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),learning_rate,training_epochs,accuracy)


    end = time.time()
    print("Total procesing time: {} seconds".format(end - begin))


if __name__ == '__main__':
    main()