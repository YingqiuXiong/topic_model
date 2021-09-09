# python3.6                                
# encoding    : utf-8 -*-                            
# @author     : YingqiuXiong
# @e-mail     : 1916728303@qq.com                                    
# @file       : corex_topic.py
# @Time       : 2021/6/28 20:17
"""
The main parameters of the CorEx topic model are:

n_hidden: number of topics ("hidden" as in "hidden latent topics")
words: words that label the columns of the doc-word matrix (optional)
docs: document labels that label the rows of the doc-word matrix (optional)
max_iter: number of iterations to run through the update equations (optional, defaults to 200)
verbose: if verbose=1, then CorEx will print the topic TCs with each iteration
seed: random number seed to use for model initialization (optional)
"""
import os

import numpy as np
import scipy.sparse as ss

import corextopic.corextopic as ct

from tqdm import tqdm


class CxTopicModel:
    def __init__(self, processed_corpus_path, outputDir, n_topic, wordSlipter, iter=200, anchor_words=None):
        self.processed_corpus_path = processed_corpus_path
        self.iter = iter
        self.n_topic = n_topic
        self.anchor_words = anchor_words
        self.wordSlipter = wordSlipter
        self.outputDir = outputDir

    # 由预处理过的语料库构建文档-词频矩阵doc_word
    def fit(self):
        # 逐行读取语料库，一行就是一篇文档
        print("--->数据文件：", self.processed_corpus_path)
        docs = []  # 文档向量集
        print("#####reading data to memory#####")
        with open(self.processed_corpus_path, 'r', encoding="gbk") as processed_corpus:
            while True:
                line = processed_corpus.readline()  # 语料库太大时，一行行读取，避免全部读入内存
                if not line:
                    break
                # docs.append(line.strip("\n").strip().split("====")[1].split(" "))
                docs.append(line.strip("\n").strip().split(self.wordSlipter))
        # 构建词典,并统计词频
        print("#####constructing vocabulary#####")
        vocab = []  # 词典
        word2id = {}  # 词到id的映射
        wordId = 0  # 词典中词的编号
        for doc_vec in tqdm(docs, ncols=100):
            for word in doc_vec:
                if not word.isdigit() and len(word) > 1:
                    if word not in vocab:
                        vocab.append(word)
                        word2id[word] = wordId
                        wordId += 1
        print("corpus size:", len(docs))
        print("vocabulary size:", len(vocab))
        # 构建文档-词频矩阵 doc_word
        print("#####constructing doc_word matrix#####")
        doc_word = np.zeros(shape=(len(docs), len(vocab)), dtype=np.int32)
        # for docId, doc in enumerate(docs, start=0):
        docId = 0
        for doc_vec in tqdm(docs, ncols=100):
            for word in doc_vec:
                if word in vocab:
                    wordId = word2id[word]
                    doc_word[docId][wordId] += 1
            docId += 1
        doc_word = ss.csr_matrix(doc_word)
        print("--->doc_word:", type(doc_word), doc_word.shape)
        print("#####traning anchored_topic_model#####")
        anchored_topic_model = ct.Corex(n_hidden=self.n_topic, seed=2, max_iter=self.iter)
        anchored_topic_model.fit(X=doc_word, words=vocab, anchors=self.anchor_words, anchor_strength=6)
        print("#####output anchored_topic#####")
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        with open(os.path.join(self.outputDir, "topic_result.txt"), "a") as f:
            f.write("#####output anchored_topic#####" + "\n\n")
            for n in range(len(self.anchor_words)):
                topic_words, _, _ = zip(*anchored_topic_model.get_topics(topic=n))
                line = str(n) + ":\t" + ', '.join(topic_words)
                f.write(line + "\n\n")
                print(line)
        print("#####output all topics from the CorEx topic model#####")
        # Print all topics from the CorEx topic model
        with open(os.path.join(self.outputDir, "topic_result.txt"), "a") as f:
            f.write("#####output all topics from the CorEx topic model#####" + "\n\n")
            topics = anchored_topic_model.get_topics()
            for n, topic in enumerate(topics):
                topic_words, _, _ = zip(*topic)  # 解压封装在一起的元组
                line = 'topic ' + str(n) + ':\t' + ', '.join(topic_words)
                f.write(line + "\n\n")
                print(line)
        print("#####output the probabilities across topics for documents#####")
        with open(os.path.join(self.outputDir, "doc_topic.txt"), "a") as f:
            doc_topic_matrix = anchored_topic_model.p_y_given_x
            for doc_prob in doc_topic_matrix:  # doc_prob 应该是一个数组，表示当前文档属于各个主题的概率
                topics = ""
                top_topic_ids = np.argsort(doc_prob)[:-11:-1]
                for top_topic_id in top_topic_ids:
                    topics += (str(top_topic_id) + ": " + str(format(doc_prob[top_topic_id], ".5f")) + "\t")
                f.write(topics + "\n")


# demo
if __name__ == '__main__':
    # 注意：corextopic每次需要修改数据文件的地址，锚定词和文档的词之间的分隔符
    # processed_corpus_path = "../data/car/rawdata_process13_chejiaohao.txt"
    processed_corpus_path = "data/oculus/reviews_processed.txt"
    K = 50
    wordSlipter = " "
    outputDir = "data/oculus/corextopic_" + str(K)
    anchor_words = [
        ["vr", "quest", "rift"],
        ["graphics", "visual", "graphic"],
        ["multiplayer", "share", "friend"],
        ["sound", "soundtrack"],
        ["developer", "update", "version"],
        ["gameplay", "mechanic"],
        ["story", "episode"],
        ["buy", "pay", "price"],
        ["control", "headset"],
        ["experience", "immersion", "immersive"],
        ["level", "difficulty", "challenge"]
    ]
    # anchor_words = [
    #     ["油耗", "省油"],
    #     ["外观", "造型", "外形"],
    #     ["操控", "控制"],
    #     ["噪音", "隔音"],
    #     ["空间", "座椅"],
    #     ["动力", "涡轮", "制动"],
    #     ["内饰", "材质"],
    #     ["续航", "里程"]
    # ]
    cxtm = CxTopicModel(processed_corpus_path=processed_corpus_path, outputDir=outputDir,
                        n_topic=K, wordSlipter=wordSlipter, iter=1000, anchor_words=anchor_words)
    cxtm.fit()

