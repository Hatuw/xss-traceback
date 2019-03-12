# coding: utf-8
import logging
import os
import re
import time
import warnings

import gensim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
)

# set global vars
data_dir = '../data'
# mirror_urls_file = 'mirror_demo.csv'  # toy dataset
mirror_urls_file = 'xss.csv'  # full dataset
less_data = 10


def list2np(x):
    array = np.array(x)
    return array


def load_data():
    """
        load mirrors file
        @return:
            pruned_data: <pandas.DataFrame>
    """
    global pruned_data
    file_path = os.path.join(data_dir, mirror_urls_file)
    print(file_path)
    assert os.path.exists(file_path), "file \"{}\" not exist".format(file_path)
    data = pd.read_csv(file_path)
    # get data columns
    columns = data.columns
    if 'Author' in columns and 'URL' in columns and 'Pagerank' in columns:
        # extract `Author` and `URL` and `Pagerank` from data
        pruned_data = data.loc[:, ['Author', 'URL', 'Pagerank']]
    else:
        exit(1)

    # 发现数据部分`URL`为nan，返回前先进行处理
    shape_bf = pruned_data.shape[0]
    print("[info] Total loaded {} lines of data.".format(shape_bf))

    pruned_data.dropna(inplace=True)
    pruned_data = pruned_data.reset_index(drop=True)  # 重新设置索引

    shape_af = pruned_data.shape[0]
    print("[info] Remove {} lines of (nan) data, Validated {} lines of data."
          .format(shape_bf - shape_af, shape_af))

    return pruned_data


def write_data(file_path_, data_split_labels):
    data_dir_tmp = "../data"
    mirror_urls_file_tmp = file_path_
    file_path = os.path.join(data_dir_tmp, mirror_urls_file_tmp)
    f = open(file_path, 'w', encoding='utf8', newline='')
    for word in data_split_labels:
        f.write(word + "\n")
    f.close()


def get_one_author_data():
    # calculate the number of author who only have one payload.
    word_ict = {}  # vocabulary
    # data_dir = "../data/data_v1"
    # one labels_map  {4: '-tmh-', 5: '..', 9: '04hrb',
    #  10: '0bj3ctiv3@gmail.com',
    #  11: '0bsolet', 12: '0c001', 15: '0rijin5l',....}
    one_dict = []
    # index = {}  # all labels
    # result = {}  # one labels index in range(45813)

    data = open("../data/vocabulary", 'r', encoding='UTF-8')
    for i in range(64461):
        word = data.readline()
        if len(word.split(" ")) == 2:
            word_ict[str(word.split(" ")[0])] = word.split(" ")[1][:-1]
        else:
            continue

    labels_map = open("../data/labels_map.csv", 'r', encoding='UTF-8')
    labels = open("../data/labels.csv", 'r', encoding='UTF-8')
    labels_data = labels.read()
    data_split_labels = labels_data.split("\n")
    del data_split_labels[-1]  # 最后一行是回车

    labels_map_data = labels_map.read()
    data_split_labels_map = labels_map_data.split("\n")
    del data_split_labels_map[-1]  # 最后一行是回车

    sum_flag = 0
    new_data_split_labels_map = []
    # print(data_split_labels_map)
    # ['-', '-Chosen-', '-[SiLeNtp0is0n]-', '-quik', '-tmh-',...]
    for word in data_split_labels_map:
        # here can change the number of the least data ...
        if int(word_ict[word]) < less_data:
            sum_flag = sum_flag + 1
            one_dict.append(word)
        else:
            new_data_split_labels_map.append(word)
    new_data_split_labels_map.append('myclassfiy')
    print(len(one_dict))  # 2058
    print(len(new_data_split_labels_map))  # 569
    # print(data_split_labels.reset_index(drop=True))
    write_data("new_labels_map.csv", new_data_split_labels_map)

    pruned_data_tmp = load_data()
    # print(pruned_data_tmp['Author'])
    pruned_author = pruned_data_tmp['Author']
    sum_flag = 0
    for i, author in enumerate(pruned_author):
        if author not in new_data_split_labels_map:
            sum_flag = sum_flag + 1
            pruned_data_tmp.iloc[i, 0] = 'myclassfiy'
            # print(i)
            # print(pruned_data_tmp['Author'][i])
    print(type(pruned_data_tmp['Author']))
    # print(sum_flag)  #3926
    # print(pruned_data_tmp.loc[:, ['Author', 'URL']])
    # print(i)  #45812
    # print(pruned_data_tmp)
    pruned_data_v1 = pruned_data_tmp.loc[:, ['Author', 'URL']]
    pruned_data_v1.to_csv("../data/data_v1/xss.csv", index=False, sep=',')
    return pruned_data_v1


def split_urls(urls, pagerank):
    """
        split urls
        params:
            urls: string[]
        @return:
            splited_urls: string[]
    """
    result_urls = []

    # remove domain
    fix_urls = [re.split('/', url, 3)[-1] for url in urls]
    for idx, url in enumerate(fix_urls):
        # split %xx
        tmp_url = re.split(r'(%\w{2})', url)
        tmp_url = [item for item in tmp_url if item]
        fix_urls[idx] = tmp_url

        # split symbol
        tmp_fixed_url = [str(pagerank[idx])]
        for item in fix_urls[idx]:
            if not re.match(r'%\w{2}', item):
                for re_result in re.split(r'(\W)', item):
                    if re_result:
                        tmp_fixed_url.append(re_result)
            else:
                tmp_fixed_url.append(item)

        result_urls.append(tmp_fixed_url)
    else:
        return result_urls


def train_d2v(urls, load=False, save_model=False, save_data=False):
    """
        trianing doc2vec
        params:
            urls: string[]
            load: boolean, load model?
            save_model: boolean, save model after training?
            save_data: boolean, save data after training?
        @return:
            encoded_urls: <pandas.DataFrame>
            model: <class 'gensim.models.doc2vec.Doc2Vec'>
    """
    data_dir_tmp = "../data"
    model_out_path = os.path.join(data_dir_tmp, "word2vec.model")
    data_out_path = os.path.join(data_dir_tmp, "train.csv")
    model = None
    if load and os.path.exists(model_out_path):
        model = gensim.models.Doc2Vec.load(model_out_path)
        print("[info] Successfully loaded word2vec model.")

    if not model:
        # if model not loaded/exist, re-train the Doc2Vec model
        # construct training data for `doc2vec`
        x_training = []
        for idx, url in enumerate(urls):
            document = gensim.models.doc2vec.TaggedDocument(url, tags=[idx])
            x_training.append(document)

        print("[info] Training word2vec model...")
        model = gensim.models.Doc2Vec(x_training, min_count=1, workers=4)
        model.train(x_training, total_examples=model.corpus_count, epochs=10)

    # transfer urls to vector
    encoded_urls = []
    for url in urls:
        encoded_urls.append(model.infer_vector(url))
    encoded_urls = pd.DataFrame(encoded_urls)

    # save model
    if save_model:
        model.save(model_out_path)
        print("[info] Export model to {}.".format(model_out_path))

    # save data
    if save_data:
        encoded_urls.to_csv(data_out_path, index=None, header=None)
        print("[info] Export url data to {}.".format(data_out_path))

    return encoded_urls, model


def encode_authors(authors, save_data=False):
    """
        encode authors by one-hot encoding
    """
    data_dir_tmp = "../data"  # change_dir
    map_out_path = os.path.join(data_dir_tmp, "labels_map.csv")
    data_out_path = os.path.join(data_dir_tmp, "labels.csv")

    encoder = LabelEncoder()
    encoder.fit(authors)  # fit model
    encoded_data = encoder.transform(authors)  # encode
    encoded_data = pd.DataFrame(encoded_data)
    if save_data:
        # save labels map(the map of number-author)
        with open(map_out_path, 'w') as f_out:
            for cls_ in encoder.classes_:
                f_out.write(cls_ + "\n")

        # save encoded data
        encoded_data.to_csv(data_out_path, index=None, header=None)
        print("[info] Export labels to {}.".format(data_out_path))

    print(encoder.classes_)
    return encoded_data


def main():
    # low_classify = 1,Reduce the number of categories
    low_classify = 0
    print("begin classify......")
    begin = time.time()

    # load data
    if low_classify == 1:
        pruned_data_tmp = get_one_author_data()
    else:

        pruned_data_tmp = load_data()

    # parse urls data
    assert 'URL' in pruned_data_tmp.columns, '`URL` must in columns'
    np.isnan(pruned_data_tmp['Pagerank'])
    pruned_urls = split_urls(pruned_data_tmp['URL'], pruned_data_tmp['Pagerank'])
    # trian word2vec
    encoded_urls, _ = train_d2v(pruned_urls,
                                load=True,
                                save_model=True,
                                save_data=True)

    # parse author data
    assert 'Author' in pruned_data_tmp.columns, '`Author` must in columns'
    authors = encode_authors(pruned_data_tmp['Author'], save_data=True)

    # del unused vars
    del encoded_urls, authors

    end = time.time()
    print("Total procesing time: {} seconds".format(end - begin))


if __name__ == '__main__':
    main()