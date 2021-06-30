import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import numpy as np
import re
from nltk import tokenize
from spacy.pipeline.pipes import to_categorical



def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def get_spilt(data_train_this, spilt):
    contents = []
    labels = []
    texts = []
    ids = []
    # np.random.seed(73)
    print(data_train_this.shape)
    for idx in range(data_train_this.content.shape[0]):
        text = BeautifulSoup(data_train_this.content[idx], features="html5lib")
        text = clean_str(str(text.get_text().encode('ascii', 'ignore')))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        contents.append(sentences)
        ids.append(data_train_this.id[idx])

        labels.append(data_train_this.label[idx])

    labels = np.asarray(labels)
    labels = to_categorical(labels)

    return train_test_split(ids, contents, labels, test_size=spilt, random_state=420,
                            stratify=labels)


def _fit_on_texts_and_comments(train_x, val_x):
    texts = []
    texts.extend(train_x)
    texts.extend(val_x)
    tokenizer = Tokenizer(num_words=20000)
    all_text = []

    all_sentences = []
    for text in texts:
        for sentence in text:
            all_sentences.append(sentence)

    all_text.extend(all_sentences)
    tokenizer.fit_on_texts(all_text)
    VOCABULARY_SIZE = len(tokenizer.word_index) + 1
    print(VOCABULARY_SIZE)


platform = 'politifact'
data_train = pd.read_csv('data/' + platform + '_content_no_ignore.tsv', sep='\t')
id_train, id_test, x_train, x_val, y_train, y_val = get_spilt(data_train, 0.25)
_fit_on_texts_and_comments(x_train, x_val)
