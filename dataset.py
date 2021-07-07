import re

import pandas as pd
from bs4 import BeautifulSoup
from nltk import tokenize
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path):
        super(MyDataset, self).__init__()

        texts, labels = [], []
        word_length_list = []
        sent_length_list = []
        with open(data_path) as csv_file:
            reader = pd.read_csv(csv_file, sep='\t')
            for idx in range(reader.content.shape[0]):
                text = ""
                content = BeautifulSoup(reader.content[idx], features="html5lib")
                content = clean_str(str(content.get_text().encode('ascii', 'ignore')))
                sentences = tokenize.sent_tokenize(content)
                sent_length_list.append(len(sentences))
                for s in sentences:
                    text += s
                    text += " "
                label = int(reader.label[idx])
                texts.append(text)
                labels.append(label)
                for sent in sentences:
                    word_list = word_tokenize(sent)
                    word_length_list.append(len(word_list))
            sorted_word_length = sorted(word_length_list)
            sorted_sent_length = sorted(sent_length_list)
        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = sorted_sent_length[int(0.8 * len(sorted_sent_length))]
        self.max_length_word = sorted_word_length[int(0.8 * len(sorted_word_length))]
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label


if __name__ == '__main__':
    test = MyDataset(data_path="../data/Constraint_content_no_ignore.tsv", dict_path="../glove.6B.100d.txt")
    print(test.__getitem__(index=1)[0].shape)
    print(test.__getitem__(index=1)[0])
