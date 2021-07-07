from collections import defaultdict

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# from MF.MF import matrix_factorization
# from readdata.rating import Rating

def reademb(file, K):
    with open(file, "r", encoding="utf-8") as f:
        f.seek(0)  # 把指针移到文件开头位置
        # user1 = []
        emb = []
        # one = True
        for line in f.readlines():  # readlines以列表输出文件内容
            # if one == True:
            #     one = False
            #     continue
            # else:
            line = line.split()
            # user1.append(line[0])
            for i in range(K):
                line[i] = float(line[i])
            emb.append(np.array(line[0:K]))
    # return user1, emb
    return emb


def readlabel(file):
    with open(file, "r", encoding="utf-8") as f:
        f.seek(0)  # 把指针移到文件开头位置
        # user = []
        label = []
        for line in f.readlines():  # readlines以列表输出文件内容
            # line = line.split()
            # user.append(line[0])
            label.append(int(line))

    return label


# 做各种嵌入分布的实验
# -------------------------------MF-----------------------------------
# R, N, M, user0, item1, MFlabel = Rating("./dataset/amazon/profiles.txt","./dataset/amazon/labels.txt")

# -----保存用户角标、商品角标-------
# u = defaultdict(dict)
# i = defaultdict(dict)
# for user in R:
#     u[user] = user0.index(user)
#     for item in R[user]:
#         i[item] = item1.index(item)

# 获取用户初代嵌入
# MFemb,nQ=matrix_factorization(R,N,M,16,u,i)
# -------------------------------MF-----------------------------------

# -------------------------------Codetector-----------------------------------
COlabel = []
COemb = reademb("do_txtfile1/sence4.txt", 128)

COlabel = readlabel("do_txtfile1/label4.txt")

# for i in range(len(user1)):
#     index = user2.index(user1[i])
#     COlabel.append(label1[index])
# for i in range(580):
#     COlabel.append(0)
# COlabel.append(1)
# -------------------------------Codetector-----------------------------------

# -------------------------------Bayes-----------------------------------
# BAlabel = []
# user4,BAemb = reademb("./ba-emb.txt",16)
# user5,label4 = readlabel("./dataset/amazon/labels.txt")
#
# for i in range(len(user4)):
#     index = user5.index(user4[i])
#     BAlabel.append(label4[index])
# -------------------------------Bayes-----------------------------------

# -------------------------------Node2vec-----------------------------------
# NOlabel = []
# user7,NOemb = reademb("./Node2vec-emb.txt",16)
# user8,label7 = readlabel("./dataset/amazon/labels.txt")
#
# for i in range(len(user7)):
#     index = user8.index(user7[i])
#     NOlabel.append(label7[index])
# -------------------------------Node2vec-----------------------------------

# tsne_mf = TSNE(n_components=2, learning_rate=100).fit_transform(MFemb)
# tsne_mf_0 = []
# tsne_mf_1 = []
# for i in range(len(MFlabel)):
#     if MFlabel[i] == int(0):
#         tsne_mf_0.append(tsne_mf[i])
#     else:
#         tsne_mf_1.append(tsne_mf[i])
# tsne_mf_0 = np.array(tsne_mf_0)
# tsne_mf_1 = np.array(tsne_mf_1)
for xxxxxx in range(1):
    tsne_co = TSNE(n_components=2, learning_rate=100).fit_transform(COemb)
    # print(len(tsne_co))
    tsne_co_0 = []
    tsne_co_1 = []
    tsne_co_2 = []
    tsne_co_3 = []
    # print(len(COlabel))
    for i in range(len(COlabel)):
        if COlabel[i] == int(0):
            tsne_co_0.append(tsne_co[i])
        elif COlabel[i] == int(1):
            tsne_co_1.append(tsne_co[i])
        elif COlabel[i] == int(2):
            tsne_co_2.append(tsne_co[i])
        else:
            tsne_co_3.append(tsne_co[i])
    start = 600
    tsne_co_0 = np.array(tsne_co_0)[start:start + 200]
    tsne_co_1 = np.array(tsne_co_1)[start:start + 200]
    tsne_co_2 = np.array(tsne_co_2)[0:200]
    tsne_co_3 = np.array(tsne_co_3)[0:200]

    # tsne_ba = TSNE(n_components=2, learning_rate=100).fit_transform(BAemb)
    # tsne_ba_0 = []
    # tsne_ba_1 = []
    # for i in range(len(BAlabel)):
    #     if BAlabel[i] == int(0):
    #         tsne_ba_0.append(tsne_ba[i])
    #     else:
    #         tsne_ba_1.append(tsne_ba[i])
    # tsne_ba_0 = np.array(tsne_ba_0)
    # tsne_ba_1 = np.array(tsne_ba_1)

    # tsne_no = TSNE(n_components=2, learning_rate=100).fit_transform(NOemb)
    # tsne_no_0 = []
    # tsne_no_1 = []
    # for i in range(len(NOlabel)):
    #     if NOlabel[i] == int(0):
    #         tsne_no_0.append(tsne_no[i])
    #     else:
    #         tsne_no_1.append(tsne_no[i])
    # tsne_no_0 = np.array(tsne_no_0)
    # tsne_no_1 = np.array(tsne_no_1)

    # 设置画布的大小
    plt.figure(figsize=(5, 5))

    # p1 = plt.subplot(141)
    # p1.set_title('MF')
    # p1.scatter(tsne_mf_0[:, 0], tsne_mf_0[:, 1], c='forestgreen', label='legitimate users')
    # p1.scatter(tsne_mf_1[:, 0], tsne_mf_1[:, 1], c='orangered', label='malicious users')
    # p1.legend(loc='upper right')

    p2 = plt.subplot()
    p2.set_title('Data Distribution With Domain Adaptation')
    p2.scatter(tsne_co_0[:, 0], tsne_co_0[:, 1], c='forestgreen', label='Source fake news', marker='^')
    p2.scatter(tsne_co_1[:, 0], tsne_co_1[:, 1], c='forestgreen', label='Source True news', marker='x')
    p2.scatter(tsne_co_2[:, 0], tsne_co_2[:, 1], c='orangered', label='Target fake news', marker='^')
    p2.scatter(tsne_co_3[:, 0], tsne_co_3[:, 1], c='orangered', label='Target True news', marker='x')
    p2.legend(loc='upper right')

    # p3 = plt.subplot(143)
    # p3.set_title('BayesDetector')
    # p3.scatter(tsne_ba_0[:, 0], tsne_ba_0[:, 1], c='forestgreen', label='legitimate users')
    # p3.scatter(tsne_ba_1[:, 0], tsne_ba_1[:, 1], c='orangered', label='malicious users')
    # p3.legend(loc='upper right')
    #
    # p4 = plt.subplot(144)
    # p4.set_title('ALDA (without DCGAN)')
    # p4.scatter(tsne_no_0[:, 0], tsne_no_0[:, 1], c='forestgreen', label='legitimate users')
    # p4.scatter(tsne_no_1[:, 0], tsne_no_1[:, 1], c='orangered', label='malicious users')
    # p4.legend(loc='upper right')

    plt.show()
