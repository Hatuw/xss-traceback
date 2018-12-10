# coding: utf-8
import os
from time import time
import pandas as pd
import multiprocessing
#import parse_data
import numpy as np
import csv
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import tensorflow as tf


# set global vars
DATA_DIR = '../data'
mirror_urls_file = 'xss.csv'
mylist = []


def init_url():
    pruned_data = parse_data.load_data()
    assert 'URL' in pruned_data.columns, 'data format error'
    pruned_data = excludenan(pruned_data)
    pruned_urls = parse_data.split_urls(pruned_data['URL'])
    return pruned_urls,pruned_data
def set_union(a,b):
    c = list(set(a + b))
    return c

def save_xss_data(pruned_urls,pruned_data):
    # save data that have divided to xss_payload.csv
    global mylist
    csvFile = open("xss_payload.csv", "w",encoding='utf8',newline='')
    writer = csv.writer(csvFile)
    for i,list_url in enumerate(pruned_urls):
        # writer.writerow("".join(pruned_data['Author'][i]))
        list_url.append(pruned_data['Author'][i])
        writer.writerow(list_url)
    csvFile.close()

    # write data to word.txt which is all of the word
    #  but I think it's useless
    # with open("word.txt", "w") as f:
    #     for pruned in pruned_urls:
    #         mylist = set_union(pruned,mylist)
    #         for word in pruned:
    #             f.write(word)
    #             f.write(' ')
    #     f.close()


def myword2vec_build(pruned_urls):
    # build the model of word2vec and save model
    model = gensim.models.Word2Vec(pruned_urls,
                                   size=50,
                                   window=10,
                                   min_count=0,
                                   workers=multiprocessing.cpu_count())

    model.save("../data/word2vec_gensim")
    model.wv.save_word2vec_format("../data/word2vec_org",
                                  "../data/vocabulary",
                                  binary=False)
    return model

def myword2vec_load():
    #load the model
    model = gensim.models.Word2Vec.load('../data/word2vec_gensim')
    return model

def excludenan(pruned_data):
    #print(np.any(pruned_data['URL'].isnull())) #是否存在nan
    pruned_data = pruned_data.dropna(axis=0,how='any') #清洗数据
    pruned_data = pruned_data.reset_index(drop=True) #重新建立索引
    return pruned_data

def readvocabulary():
    # read the vocabulary....
    # It doesn't work right now
    f = open('../data/vocabulary', 'r')
    tmp = f.read().split('\n')
    c = []
    for i in tmp:
        c.append(i.split(" ")[0])
    return  c

def readpayload(maxlen=999999,flag = 0):
    # read urls & author from csv
    f = open('./xss_payload.csv', 'r',encoding='UTF-8')
    result = {}
    reader = csv.reader(f)
    author = []
    for item in enumerate(reader):
        if(item[0]>=maxlen):
            break
        tmp = item[1]
        if(len(tmp) == 0):
            print("something wrong......")
        author.append(tmp[-1])
        result[item[0]] = item[1]
        if(flag == 1):             #flag等于1的时候不取作者名字
            result[item[0]] = tmp[:-1]
    # print(len(result))
    f.close()
    return  result,author




def calculate_len(pruned_urls):
    # for calculate the avg len of url
    maxlen = 0
    minlen = 37
    # print(pruned_urls)
    print(type(minlen))
    total_len = 0
    for i in range(len(pruned_urls)):
        # print(type(len(pruned_urls[i])))
        if(len(pruned_urls[i]) > maxlen):
            maxlen,max_index = max(len(pruned_urls[i]),maxlen),i

        if(len(pruned_urls[i]) < minlen):
            minlen,min_index = min(len(pruned_urls[i]),99),i
        total_len += len(pruned_urls[i])
    print("max_url_len = " + str(maxlen))
    print("max_url_index = " + str(max_index))
    print(pruned_urls[max_index])
    print("min_url_len = " + str(minlen))
    print("min_url_index = " + str(min_index))
    print(pruned_urls[min_index])
    print("avg_url_len = " + str(total_len/len(pruned_urls)))

def save_data_txt(i,content):
    path = "../data/fix_data_2/"
    filename = "fixed_" + str(i) + '.txt'
    f = open(path + filename, 'w',encoding='UTF-8')
    writer= csv.writer(f)
    writer.writerow(content)
    f.close()

def myone_hot(labels):
    labels = set(labels)
    print("labels len = " + str(len(labels)))
    labels_key ={}
    for i,value in enumerate(labels):
        labels_key[value] = np.eye(len(labels))[i]
    return labels_key

def getVecsByWord2Vec(model, corpus,author,size,maxlen): #size是向量大小，maxlen是url长度
    # getVecsByWord2Vec and url -->size*(maxlen+1)
    # save to txt
    f = open("../data/fixed_data.csv", 'w',encoding='utf8',newline='')
    writer= csv.writer(f)
    print(len(corpus))
    # author_one_hot = myone_hot(author)
    for i in range(len(corpus)):
        url_encode = np.array([])
        j = 0
        for j,word in enumerate(corpus[i]):
            if(j>=maxlen):
                break
            try:
                url_encode = np.append(model[word], url_encode, axis=0)
            except KeyError:
                continue
        if(j<maxlen):
            for con in range(maxlen-j-1):
                url_encode = np.append([0]*size, url_encode, axis=0)
        if(url_encode.shape[0] != size*maxlen):
            #print("getVecsByWord2Vec:somethin wrong...")
            url_encode = np.append([0]*size, url_encode, axis=0)
            if(url_encode.shape[0] != size*maxlen):
                print("getVecsByWord2Vec:somethin wrong...")
        if(i%1000 == 0):
            print(str(i)+" round")
        url_encode = np.append(model[author[i]], url_encode, axis=0)
        #save_data_txt(i,url_encode)
        # print(len(model[author[i]]))
        writer.writerow(url_encode)
    f.close()

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

def mytf_idf_lsi():
    #now we can calculate the TF-IDF
    # 注意这两个都是稀疏矩阵的表示形式......lsi_matrix = lsi_sparse_matrix.toarray() 可以用这个来进行转换：lsi_sparse_matrix
    pruned_urls, author = readpayload(5000)
    pruned_urls_values = list(pruned_urls.values())
    dictionary = gensim.corpora.Dictionary(pruned_urls_values)
    # dictionary.save('../data/xss_payload.dict')
    # print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in pruned_urls_values]
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi_model = gensim.models.LsiModel(corpus=corpus_tfidf,
                                id2word=dictionary,
                                num_topics=50)
    corpus_lsi = [lsi_model[doc] for doc in corpus]
    # print(corpus_lsi)
    # for item in corpus_tfidf:
    #     print(item)
    # tfidf.save("../data/data.tfidf")
    #tfidf = gensim.models.TfidfModel.load("data.tfidf")
    return corpus_tfidf

def main():
    # load data
    #step 1 分词并存储数据 :
    # init_url() & save_xss_data(pruned_urls, pruned_data)
    #step 2 读取分词后的数据进行向量化训练并存储 :
    # pruned_urls,author = readpayload(),
    # model = myword2vec_build(list(pruned_urls.values()))
    #step 3 读取模型进行归一化处理（每个url整合为50x50=2500+50大小的向量，其中不够长的向量用0填充，超长的砍掉。
    # 2500+50：50为每个词向量的大小，50为每个url长度的规定大小(平均每个向量的长度约为37)，最后还有50为作者的向量） :
    # model = myword2vec_load()
    # getVecsByWord2Vec(model,pruned_urls,author,50,50)  #avge_len = 37

    global mylist
    begin = time()
    # if len(sys.argv) != 2:
    #     print("Please use python train_with_gensim.py mode_num")
    #     exit()
    # mode_num = sys.argv[1]
    mode_num = 3  #choose mode
    if(mode_num == 1):
        pruned_urls, pruned_data = init_url() #init_url()-->load_data() &  split_urls(urls)
        #pruned_data = load_data()  #DATA_DIR & mirror_urls_file
        #split_urls(urls) # url-->divide to word,return pruned_urls
        save_xss_data(pruned_urls, pruned_data)

    if(mode_num == 2):
        pruned_urls, author = readpayload()
        model = myword2vec_build(list(pruned_urls.values()))

    if (mode_num == 3):
        pruned_urls, author = readpayload()
        model = myword2vec_load()
        print(author)
        print(type(author))
        getVecsByWord2Vec(model,pruned_urls,author,50,50)  #avge_len = 37
    # #fixed_data = load_fixed_data(5000)
    # corpus_tfidf = mytf_idf_lsi()

    end = time()
    print("Total procesing time: %d seconds" % (end - begin))

if __name__ == '__main__':
    main()