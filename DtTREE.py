from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
import pandas as pd

pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)
feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', '稀释葡萄酒', '脯氨酸']

import graphviz
dot_data = tree.export_graphviz(clf
                                , feature_names=feature_name
                                , class_names=["琴酒", "雪莉", "贝尔摩德"]
                                , filled=True
                                , rounded=True
                                )
graph = graphviz.Source(dot_data)
print(graph)
print([*zip(feature_name, clf.feature_importances_)])
