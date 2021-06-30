import pandas as pd
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import tree, svm
import numpy as np
import itertools

stops = set(stopwords.words("english"))


def cleantext(string):
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+", ' ', text)
    text = re.sub(r"www(\S)+", ' ', text)
    text = re.sub(r"&", ' and ', text)
    tx = text.replace('&amp', ' ')
    text = re.sub(r"[^0-9a-zA-Z]+", ' ', text)
    text = text.split()
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label')
    plt.show()


def print_metrices(pred, true):
    print(confusion_matrix(true, pred))
    print(classification_report(true, pred, ))
    print("Accuracy :{:.4f} ".format(accuracy_score(pred, true)))
    print("Precison : {:.4f}".format(precision_score(pred, true)))
    print("Recall : {:.4f}".format(recall_score(pred, true)))
    print("F1 : {:.4f}".format(f1_score(pred, true)))


def cleantext(string):
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+", ' ', text)
    text = re.sub(r"www(\S)+", ' ', text)
    text = re.sub(r"&", ' and ', text)
    tx = text.replace('&amp', ' ')
    text = re.sub(r"[^0-9a-zA-Z]+", ' ', text)
    text = text.split()
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


# gossipcop politifact Constraint
trainxx = pd.read_csv('data/politifact_content_no_ignore.tsv', sep='\t')
trainxx['content'] = trainxx['content'].map(lambda x: cleantext(x))
# val = pd.read_excel('../english_val.xlsx')
# test = pd.read_excel('../english_test_labels.xlsx')

train, test = train_test_split(trainxx, test_size=0.25, random_state=42)
# pipeline = Pipeline([
#     ('bow', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('c', LinearSVC())
# ])
# fit = pipeline.fit(train['content'], train['label'])
# print('SVM')
# print('test:')
# pred = pipeline.predict(test['content'])
# print_metrices(pred, test['label'])

# plot_confusion_matrix(confusion_matrix(test['label'], pred), target_names=['fake', 'real'], normalize=False,
#                       title='Confusion matix on COVID_75')

# val_ori = pd.read_excel('../english_val.xlsx')
# svm_val_misclass_df = val_ori[pred!=val['label']]


pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('c', BernoulliNB())
])
fit = pipeline.fit(train['content'], train['label'])
print('BernoulliNB')
print('test:')
pred = pipeline.predict(test['content'])

print_metrices(pred, test['label'])
# plot_confusion_matrix(confusion_matrix(test['label'], pred), target_names=['real', 'fake'], normalize=False,
#                       title='Confusion matix of LR on val data')
# pipeline = Pipeline([
#     ('bow', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('c', tree.DecisionTreeClassifier())
# ])
# fit = pipeline.fit(train['content'], train['label'])
# print('Decision Tree')
# print('test:')
# pred = pipeline.predict(test['content'])
#
# print_metrices(pred, test['label'])
# plot_confusion_matrix(confusion_matrix(test['label'], pred), target_names=['fake', 'real'], normalize=False, \
#                       title='Confusion matix of DT on val data')
#
# pipeline = Pipeline([
#     ('bow', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('c', GradientBoostingClassifier())
# ])
# fit = pipeline.fit(train['content'], train['label'])
# print('Gradient Boost')
# print('test:')
# pred = pipeline.predict(test['content'])
#
# print_metrices(pred, test['label'])
# plot_confusion_matrix(confusion_matrix(test['label'], pred), target_names=['fake', 'real'], normalize=False, \
#                       title='Confusion matix of GDBT on val data')


# pipeline = Pipeline([
#     ('bow', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('c', AdaBoostClassifier(n_estimators=100))
# ])
# fit = pipeline.fit(train['content'], train['label'])
# print('AdaBoostClassifier')
# print('test:')
# pred = pipeline.predict(test['content'])

# print_metrices(pred, test['label'])
# plot_confusion_matrix(confusion_matrix(test['label'], pred), target_names=['fake', 'real'], normalize=False, \
#                       title='Confusion matix of GDBT on val data')
