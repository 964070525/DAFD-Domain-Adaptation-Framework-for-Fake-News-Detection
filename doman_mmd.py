import copy

import numpy as np
from sklearn import metrics
from torch import nn, optim

from dataset_no_stopwords import *
from mmd import mmd_linear
from sklearn.datasets import load_digits
from torchtext import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

# gossipcop politifact
data1 = get_Dataset("gossipcop")
data2 = get_Dataset("politifact")
all_data = get_Dataset("gossipcop", "politifact")
train_iterator, valid_iterator = data.BucketIterator.splits(
    (data1, data2),
    batch_size=415,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True)
build_vocab(data1)
sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 2)
X, y = load_digits(return_X_y=True)

tsne = TSNE()


def this(X, Y):
    delta = torch.mean(X, 0) - torch.mean(Y, 0)
    return torch.matmul(delta, torch.transpose(delta, 1, 0))


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]领域适配MMD距离
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    X1 = copy.copy(X).detach().numpy()
    Y1 = copy.copy(Y).detach().numpy()
    XX = metrics.pairwise.rbf_kernel(X1, X1, gamma)
    YY = metrics.pairwise.rbf_kernel(Y1, Y1, gamma)
    XY = metrics.pairwise.rbf_kernel(X1, Y1, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


class MMD(nn.Module):

    # 定义所有层
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, text, text2):
        embedded = self.embedding(text).detach().numpy()
        embedded2 = self.embedding(text2).detach().numpy()
        print(embedded.shape)
        print(embedded2.shape)

        for i in embedded2[0:3]:
            y = torch.zeros(len(i))
            y[0] = 1
            X_embedded = tsne.fit_transform(i)
            print("i.shape", i.shape)
            sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
        for i in embedded[0:3]:
            y = torch.ones(len(i))
            X_embedded = tsne.fit_transform(i)
            y[0] = 0
            print("i.shape", i.shape)
            sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)

        plt.show()
        return 5


model = MMD(len(TEXT.vocab), 100)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.011)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
loss_func = nn.CrossEntropyLoss()
iter1 = iter(valid_iterator)
total_mmd = []
for batch in train_iterator:
    text, text_lengths = batch.text
    if len(text_lengths) != 415:
        break
    batch1 = iter1.__next__()
    text2, text2_lengths = batch1.text
    # 转换成一维张量
    # print(text.shape)
    mmd_loss = model(text, text2)

    # 反向传播损耗并计算梯度
    loss_total = mmd_loss
    # loss_total.backward()
    total_mmd.append(mmd_loss)
    # 更新权重
    optimizer.step()
    print(mmd_loss)
    if len(text2_lengths) != 64:
        iter1 = iter(valid_iterator)
print("mean = ", np.mean(total_mmd))
