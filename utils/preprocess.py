# encoding=utf-8

import jieba

import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class TextPreProcess(object):
    def __init__(self, stopwords_path, raw_data_path, store_data_path, isLongText=True):
        """
        中英文文本预处理
        :param stopwords_path: 用户构建的停用词存储地址
        :param raw_data_path: 待处理数据的地址
        :param store_data_path: 处理后的数据存储地址
        :param isLongText: 是否是长文本，默认为长文本
        """
        self.stopwords_path = stopwords_path
        self.raw_data_path = raw_data_path
        self.store_data_path = store_data_path
        self.isLongText = isLongText

    def preprocess_cn(self):
        """
        用于中文文档预处理
        主要包括去标点，去数字，分词和去除停用词，如果是长文本只保留词数量在10个以上的文档
        """
        # 第一步，创建停用词表
        stopwords = [line.strip("\n").strip() for line in open(self.stopwords_path, 'r', encoding="gbk")]
        # 第二步，分词并且去除停用词
        with open(self.store_data_path, "a", encoding="gbk") as f:
            with open(self.raw_data_path, 'r', encoding="gbk") as raw_data:
                while True:
                    line = raw_data.readline()
                    if not line:
                        break
                    line = line.strip("\n").strip()  # 去除两头的空格和换行符
                    for c in string.punctuation:  # 去标点
                        line = line.replace(c, ' ')
                    for c in string.digits:  # 去数字
                        line = line.replace(c, '')
                    word_list = jieba.cut(line.strip())
                    doc = ''
                    for word in word_list:
                        if word in stopwords:
                            word_list.remove(word)  # 移除停用词
                            # doc += (word + ' ')
                    # 该文档有用的条件：短文本 或者 长文本但词的数量大于10
                    if not self.isLongText or len(word_list) > 10:
                        for word in word_list:
                            doc += (word + ' ')
                        f.write(doc + "\n")

    def preprocess_en(self):
        """
        用于英文文本预处理，主要包括:
        1.去标点，去数字
        2.分割成单词
        3.转小写，词形还原
        4.去除停用词和非英文单词
        5.长文本则保留单词数在10个以上的文档
        """
        # 第一步，创建停用词表
        stopwords = [line.strip("\n").strip() for line in open(self.stopwords_path, 'r').readlines()]
        # 第二步，逐行读取数据并处理后存储
        nltk.download('wordnet')
        wnl = WordNetLemmatizer()  # 词形还原对象
        with open(self.store_data_path, "a", encoding="utf-8") as f:
            with open(self.raw_data_path, 'r', encoding="utf-8") as raw_data:
                while True:
                    line = raw_data.readline()
                    if not line:  # 文件读完就跳出循环
                        break
                    doc = line.strip("\n").strip()
                    for c in string.punctuation:  # 去标点
                        doc = doc.replace(c, ' ')
                    for c in string.digits:  # 去数字
                        doc = doc.replace(c, '')
                    word_list = nltk.word_tokenize(doc)  # 分割成单词
                    cleanDoc = ""
                    new_word_list = []
                    for word in word_list:
                        # 去除停用词和非英文单词(利用wordnet)
                        word = wnl.lemmatize(word.lower())
                        if word not in stopwords and wordnet.synsets(word):
                            new_word_list.append(word)
                    # 该文档有用的条件：短文本 或者 长文本但词的数量大于10
                    if not self.isLongText or len(new_word_list) > 10:
                        for word in new_word_list:
                            cleanDoc += (word + ' ')
                        f.write(cleanDoc + "\n")


# demo
if __name__ == '__main__':
    print("demo")

