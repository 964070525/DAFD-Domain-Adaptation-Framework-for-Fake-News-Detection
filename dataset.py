import pandas as pd
import torch
from bs4 import BeautifulSoup
from nltk import tokenize
from nltk.corpus import stopwords
from numpy import long

from tqdm import tqdm
# 处理数据
from torchtext import data
import re


def tokenizer(text):  # 可以自己定义分词器，比如jieba分词。也可以在里面添加数据清洗工作
    "分词操作，可以用jieba"
    sentences = tokenize.sent_tokenize(text)
    return [item.split() for item in sentences]


TEXT = data.Field(tokenize="spacy", batch_first=True, include_lengths=True, tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float, batch_first=True, use_vocab=False)

example_total = []


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


class MyDataset(data.Dataset):

    def __init__(self, text_field, label_field, name, name2=None, **kwargs):
        global example_total
        fields = [(None, None), ('label', label_field), ('text', text_field)]
        examples = []
        if name2 is not None:
            # csv_data = pd.read_csv('data/' + name2 + '_content_no_ignore.tsv', sep='\t')
            # for label, comment in tqdm(zip(csv_data['label'], csv_data['content'])):
            #     label = long(label)
            #     comment = BeautifulSoup(comment, features="html5lib")
            #     comment = clean_str(str(comment.get_text().encode('ascii', 'ignore')))
            #     examples.append(data.Example.fromlist([None, label, comment], fields))
            super(MyDataset, self).__init__(example_total, fields, **kwargs)
            return
        csv_data = pd.read_csv('data/' + name + '_content_no_ignore.tsv', sep='\t')
        for label, comment in tqdm(zip(csv_data['label'], csv_data['content'])):
            label = long(label)
            comment = BeautifulSoup(comment, features="html5lib")
            comment = clean_str(str(comment.get_text().encode('ascii', 'ignore')))
            examples.append(data.Example.fromlist([None, label, comment], fields))
            example_total.append(data.Example.fromlist([None, label, comment], fields))
        super(MyDataset, self).__init__(examples, fields, **kwargs)


# training_data = data.TabularDataset(path='data/politifact_content_no_ignore.tsv', format='tsv', fields=fields,
#                                     skip_header=True)

def get_Dataset(name, name2=None):
    return MyDataset(text_field=TEXT, label_field=LABEL, name=name, name2=name2)


def build_vocab(train_data):
    TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")
    # LABEL.build_vocab(train_data)

    # print("Size of TEXT vocabulary:", len(TEXT.vocab))
    # print("Size of LABEL vocabulary:", len(LABEL.vocab))
    # print(LABEL.vocab.itos)
    # print(TEXT.vocab.freqs.most_common(10))
    # print(TEXT.vocab.stoi)
