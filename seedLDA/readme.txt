这个是封装的guidedLDA,方便直接使用而不需要看更多得教程
封装文档主题模型，每次只需要根据你自己的实际情况修改main方法里面的下列参数：
    processed_corpus: 分词，去停用词等预处理后的语料库文件(一条文档一行)
    K: 主题数
    n_top_words: 每个主题下的主题词的数量
    iterationNum: 迭代次数
    seed_topic_list: 种子词表
    outputDir: 存储结果的文件夹
