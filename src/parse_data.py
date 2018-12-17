# coding: utf-8
import os
import re
import gensim
from itertools import chain
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# set global vars
DATA_DIR = '../data'
# mirror_urls_file = 'mirror_demo.csv'  # toy dataset
mirror_urls_file = 'xss.csv'  # full dataset


def load_data():
    """
        load mirrors file
        @return:
            pruned_data: <pandas.DataFrame>
    """
    file_path = os.path.join(DATA_DIR, mirror_urls_file)
    assert os.path.exists(file_path), "file \"{}\" not exist".format(file_path)
    data = pd.read_csv(file_path)
    # get data columns
    columns = data.columns
    if 'Author' in columns and 'URL' in columns:
        # extract `Author` and `URL` from data
        pruned_data = data.loc[:, ['Author', 'URL']]
    else:
        exit(1)

    # 发现数据部分`URL`为nan，返回前先进行处理
    shape_bf = pruned_data.shape[0]
    print("[info] Total loaded {} lines of data.".format(shape_bf))

    pruned_data.dropna(inplace=True)

    shape_af = pruned_data.shape[0]
    print("[info] Remove {} lines of (nan) data, Validated {} lines of data."
          .format(shape_bf - shape_af, shape_af))

    return pruned_data


def split_urls(urls):
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
        tmp_fixed_url = []
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

    model_out_path = os.path.join(DATA_DIR, "word2vec.model")
    data_out_path = os.path.join(DATA_DIR, "train.csv")
    model = None
    if load and os.path.exists(model_out_path):
        model = gensim.models.Doc2Vec.load(model_out_path)
        print("[info] Successfully loaded word2vec model.")

    if not model:
        # if model not loaded/exist, re-train the Doc2Vec model
        # construct training data for `doc2vec`
        X_training = []
        for idx, url in enumerate(urls):
            document = gensim.models.doc2vec.TaggedDocument(url, tags=[idx])
            X_training.append(document)

        print("[info] Training word2vec model...")
        model = gensim.models.Doc2Vec(X_training, min_count=1, workers=4)
        model.train(X_training, total_examples=model.corpus_count, epochs=10)

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
    map_out_path = os.path.join(DATA_DIR, "labels_map.csv")
    data_out_path = os.path.join(DATA_DIR, "labels.csv")

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

    return encoded_data


# load, parse and export word vectors
def main():
    # load data
    pruned_data = load_data()

    # parse urls data
    assert 'URL' in pruned_data.columns, '`URL` must in columns'
    pruned_urls = split_urls(pruned_data['URL'])

    # trian word2vec
    encoded_urls, _ = train_d2v(pruned_urls,
                                load=True,
                                save_model=True,
                                save_data=True)

    # parse author data
    assert 'Author' in pruned_data.columns, '`Author` must in columns'
    authors = encode_authors(pruned_data['Author'], save_data=True)

    # del unused vars
    del encoded_urls, authors


if __name__ == '__main__':
    main()
