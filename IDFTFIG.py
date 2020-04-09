# coding utf-8

import os
import numpy as np
import tensorflow as tf
import data_handle as gr

# word list为词表, word dict为词与维度对应的字典, number dict为维度与词对应的字典, n_class为总的维度数目
cur = 0
word_list = []
num_classes = 9
for i in gr.all_sec:
    for j in i:
        word_list.append(j)
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)  # number of Vocabulary
res_pro = 0
# 开始十折交叉循环
for cur in range(10):
    print('第', cur + 1, '轮')
    # 提取当前的测试集与训练集
    sec = gr.sec[cur]
    test_sec = gr.test_sec[cur]
    # IDF是第二个权重, 反文档频率
    IDF = np.zeros(shape=n_class, dtype=np.float)
    # IG是第三个权重, 信息熵权益
    IG = np.zeros(shape=n_class, dtype=np.float)
    # 构造四个训练以及验证集合
    # inputs是记录训练集文本向量的不为零维度的集合，之后还要处理为input_vec才能得到文本的特征向量
    # in_te是训练集标签集合
    inputs = []
    in_te = []
    for sen in sec:
        tmp = np.asarray([word_dict[n] for n in sen])
        inputs.append(tmp)
        tmp = set(tmp)
        # 计算IDF
        for i in tmp:
            IDF[i] += 1
    for i in range(gr.total):
        in_te.append(gr.getindex(i, gr.SIZE))
    # test_inputs是记录测试集文本向量的不为零维度的集合，之后还要处理为test_vec才能得到文本的特征向量
    # test_in_te是测试集标签集合
    test_inputs = []
    test_te = []
    for sen in test_sec:
        test_inputs.append(np.asarray([word_dict[n] for n in sen]))
    for i in range(gr.test_total):
        test_te.append(gr.getindex(i, gr.test_SIZE))
    # input_vec是长度为n_class的文本向量(训练集)
    # test_vec同上(为测试集)
    input_vec = []
    test_vec = []
    for i in inputs:
        tmpe = np.zeros(shape=n_class, dtype=np.float32)
        for j in i:
            tmpe[j] += 1
        input_vec.append(tmpe)
    for i in test_inputs:
        tmpe = np.zeros(shape=n_class, dtype=np.float32)
        for j in i:
            tmpe[j] += 1
        test_vec.append(tmpe)
    # text_pro各个文本类别的先验概率
    text_pro = np.zeros(shape=9, dtype=np.float)
    for i in range(9):
        text_pro[i] = 10000 * (gr.SIZE[i] + 1) / (gr.total + 9)
        #text_pro[i] = 1 / text_pro[i]
    text_pro = np.log2(text_pro)
    # 更改SIZE的设定，变为前k和
    SIZE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(9):
        SIZE[i] = gr.SIZE[i]
    for i in range(1, 9):
        SIZE[i] += SIZE[i - 1]
    # input1是第一步处理数据，压缩成各个类别词出现的频数(为9*n_class维)
    input1 = []
    for i in range(9):
        input1.append(np.sum(input_vec[SIZE[i - 1]:SIZE[i]], 0))
    # 分别将input1横向的与列向的相加
    in_sum = np.sum(input1, 1)
    in_sum2 = np.sum(input1, 0)
    # 求出IDF
    for i in range(n_class):
        if IDF[i] != 0:
            IDF[i] = np.log2(gr.total / IDF[i] + 0.01)
        else:
            IDF[i] = -1
    # 求出第三个权重IG
    probi1 = np.zeros(shape=9, dtype=np.float)
    for i in range(9):
        tmpeint = gr.SIZE[i] / gr.total
        probi1[i] = tmpeint
    pro_2 = 0
    for j in range(9):
        pro_2 += - probi1[j] * np.log10(probi1[j])
    for i in range(n_class):
        p2 = np.zeros(shape=9, dtype=np.float)
        for j in range(9):
            p2[j] = (input1[j][i] + 1) / (in_sum2[i] + 2)
        for j in range(9):
            IG[i] += p2[j] * np.log10(p2[j])
        IG[i] += pro_2 + 1
    # input3为权重IG与IDF的点乘
    input3 = np.zeros(shape=n_class, dtype=np.float)
    input3 = np.multiply(IG, IDF)
    # input2为各个词语的条件后验概率, 用log2处理
    input2 = np.zeros(shape=[9, n_class], dtype=np.float)
    for i in range(9):
        for j in range(n_class):
            input2[i][j] = (input1[i][j] + 1) / (in_sum[i] + n_class)
    input2 = np.log2(input2)
    # 将后验概率的log与权重相乘
    input3 = np.multiply(input2, input3)
    # 循环测试集
    epoch = 0
    c = 0
    w = 0
    for i in range(9):
        # 每一类中测试集的正确错误数目
        correct_num = 0
        wrong_num = 0
        for j in range(gr.test_SIZE[i]):
            # input3乘TF权重
            z = np.multiply(input3, test_vec[epoch])
            # 降维
            t_z = np.sum(z, 1)
            # 加上后验概率
            res = np.add(t_z, text_pro)
            # 求出预测的标签类别
            index = np.argmax(res, 0)
            if index == test_te[epoch]:
                correct_num += 1
                c += 1
            else:
                wrong_num += 1
            epoch += 1
        print(correct_num / gr.test_SIZE[i])
    # 一轮的平均正确率
    print("总的正确率:", c / gr.test_total)
    res_pro += c / gr.test_total
print("十折交叉循环下的平均正确率:", res_pro / 10)

