# encoding = utf-8
# python3.6
# encoding    : utf-8 -*-
# @author     : YingqiuXiong
# @e-mail     : 1916728303@qq.com
# @file       : 1.py
# @Time       : 2021/5/28 11:25

import os

import guidedlda
import numpy as np

from tqdm import tqdm


class SeedLda:
    def __init__(self, processed_corpus, K, seed_topic_list, outputDir, n_top_words, wordSpliter: str, iterationNum=100):
        """
        封装guidedlda
        :param processed_corpus: 预处理过的语料库地址
        :param K: 主题数
        :param seed_topic_list: 种子词列表
        :param iterationNum: 迭代次数
        :param n_top_words: 主题代表词的数量
        :param outputDir: 结果输出目录
        :param wordSpliter: 一行文档中词与词之间的分隔符
        """
        self.processed_corpus = processed_corpus
        self.K = K
        self.seed_topic_list = seed_topic_list
        self.outputDir = outputDir
        self.n_top_words = n_top_words
        self.wordSpliter = wordSpliter
        self.iterationNum = iterationNum

    def seedLda(self):
        print("--->数据文件的地址:", self.processed_corpus)
        print('#####loading data file#####')
        documents = []
        with open(self.processed_corpus, "r", encoding="gbk") as corpus:
            while True:
                line = corpus.readline()
                if not line:
                    break
                doc_vec = line.strip('\n').strip("\t").strip().split(self.wordSpliter)
                documents.append(doc_vec)
        print('------>corpus size:', len(documents))
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        # 第一步，构建词典
        print('####constructing dictionary######')
        vocab = []
        if os.path.exists(os.path.join(self.outputDir, "vocab.txt")):  # 词典已经存在
            with open(os.path.join(self.outputDir, "vocab.txt"), 'r') as v:
                while True:
                    line = v.readline()
                    if not line:  # 读没有了
                        break
                    vocab.append(line.strip("\n").strip().split("\t")[1])
        else:
            for doc in tqdm(documents, ncols=100):
                for word in doc:
                    if word not in vocab:
                        vocab.append(word)
            print('####store dictionary######')
            with open(os.path.join(self.outputDir, "vocab.txt"), 'a') as v:
                for wordId, word in enumerate(vocab, start=0):
                    v.write(str(wordId) + ":" + "\t" + word + "\n")
        print('------>vocab size:', len(vocab))
        # 第二步，构建有索引的词典
        word2id = dict((verb, verb_id) for verb_id, verb in enumerate(vocab, start=0))
        # 第三步，构建 文档—词 矩阵表示
        # 构建 M*V 全零矩阵，文档中的词频
        doc_word = np.zeros(shape=(len(documents), len(vocab)), dtype=np.int32)
        # 赋值
        print('####constructing doc_word matrix######')
        for m, doc in enumerate(documents, start=0):
            for word in doc:
                word_id = word2id[word]
                doc_word[m][word_id] += 1
        # 第四步，构建种子词汇
        print('####constructing seed_topics######')
        seed_topics = {}
        for topic_id, topic_seeds in enumerate(self.seed_topic_list, start=0):
            for word in topic_seeds:
                try:
                    seed_topics[word2id[word]] = topic_id
                except Exception:
                    print("seed word {} not in vocabulary".format(word))
                    continue
        # 第五步，训练模型
        print('########model training#########')
        model = guidedlda.GuidedLDA(n_topics=self.K, n_iter=self.iterationNum, random_state=7, refresh=20)
        model.fit(doc_word, seed_topics=seed_topics, seed_confidence=0.15)
        print('########storing model#########')
        # 第六步，输出每个主题下的词
        print('########output topic distribution#########')
        with open(os.path.join(self.outputDir, "topic_word.txt"), "a", encoding="utf-8") as f:
            topic_word = model.topic_word_  # 主题-词分布矩阵
            for i, topic_dist in enumerate(topic_word, start=0):
                top_n_word_id = np.argsort(topic_dist)[: -(self.n_top_words + 1): -1]
                line = "Topic " + str(i) + ":" + "\n"
                for word_id in top_n_word_id:
                    line += (vocab[word_id] + ": " + str(format(topic_dist[word_id], ".5f")) + "\n")
                f.write(line + "\n\n")
        # 输出文档下的主题分布
        print('########output document distribution#########')
        with open(os.path.join(self.outputDir, "doc_topic.txt"), "a", encoding="utf-8") as f:
            doc_topic_dist = model.transform(doc_word)  # 文档-主题分布矩阵
            for doc_id, current_prob in enumerate(doc_topic_dist, start=0):
                line = "Document " + str(doc_id) + ":"
                top_topic_ids = np.argsort(current_prob)[:-11:-1]
                for top_topic_id in top_topic_ids:
                    line += ("\t" + str(top_topic_id) + ": " + str(format(current_prob[top_topic_id], ".5f")))
                f.write(line + "\n")


# demo
if __name__ == '__main__':
    """
    封装文档主题模型，每次只需要修改main方法里面的下列参数：
    processed_corpus: 分词，去停用词等预处理后的语料库文件(一条文档一行)
    K: 主题数
    n_top_words: 每个主题下的主题词的数量
    iterationNum: 迭代次数
    seed_topic_list: 种子词表
    outputDir: 存储结果的文件夹
    """
    processed_corpus = "data/oculus/reviews_processed.txt"
    K = 50
    n_top_words = 15
    iterationNum = 1000
    seed_topic_list = [
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
    outputDir = "data/oculus/seedlda_" + str(K)
    seedLda = SeedLda(processed_corpus, K, seed_topic_list, outputDir, n_top_words,
                      wordSpliter=" ", iterationNum=iterationNum)
    seedLda.seedLda()
