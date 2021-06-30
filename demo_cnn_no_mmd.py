import random

from torch import optim, nn
import numpy as np

from TextCNN import CNN
from dataset import *
from model import classifier
from train_model import *
import os
import matplotlib.pyplot as plt

from util import change_dataset, change_dataset_25, change_dataset_50, change_dataset_75

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
SEED = 2019
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
dataset_name = "gossipcop"
# gossipcop politifact Constraint
source_data = get_Dataset(dataset_name)
# print(vars(source_data.examples[0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build_vocab(source_data)
# 设置batch大小
BATCH_SIZE = 16
train_data, valid_data = source_data.split(split_ratio=0.05, random_state=random.seed(2019))
total_valid_loss = []
total_valid_acc = []
total_valid_recall = []
total_valid_f1 = []
total_valid_precision = []
total_valid_valid_auc = []


def change_evaluating(i, loss, acc, recall, f1, precision, auc):
    if len(total_valid_f1) == i:
        total_valid_loss.append(loss)
        total_valid_acc.append(acc)
        total_valid_recall.append(recall)
        total_valid_f1.append(f1)
        total_valid_precision.append(precision)
        total_valid_valid_auc.append(auc)
    else:
        if f1 > total_valid_f1[i]:
            total_valid_loss[i] = loss
            total_valid_acc[i] = acc
            total_valid_recall[i] = recall
            total_valid_f1[i] = f1
            total_valid_precision[i] = precision
            total_valid_valid_auc[i] = auc


for i in range(0, 1):
    # train_data, valid_data = change_dataset_50(i, train_data, valid_data)
    # 载入迭代器
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)
    train_iterator1 = None
    size_of_vocab = len(TEXT.vocab)
    embedding_dim = 100
    num_hidden_nodes = 50
    num_output_nodes = 2
    num_layers = 2
    bidirection = True
    dropout = 0.2
    # 实例化模型
    model = CNN(size_of_vocab, embedding_dim, 0.2)
    # model = HAN_Attention(size_of_vocab, embedding_dim, 50, 2)
    print(model)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(model):,} trainable parameters')

    # 初始化预训练embedding
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    print(pretrained_embeddings.shape)

    # 定义优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=0.0051)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_func = nn.CrossEntropyLoss()

    # 如果cuda可用
    model = model.to(device)
    criterion = loss_func.to(device)

    N_EPOCHS = 20
    best_valid_loss = float('inf')
    print("train on", len(train_data), "    Valid on", len(valid_data))
    lr_list = []

    for epoch in range(N_EPOCHS):
        # 训练模型
        train_loss, train_acc, train_recall, train_f1, train_precision, train_auc = train_without_mmd(
            model,
            train_iterator,
            optimizer,
            criterion)
        # 评估模型
        valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_auc = evaluate(model, valid_iterator,
                                                                                             criterion)
        change_evaluating(i, valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_auc)
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        print(
            "epoch: ", epoch + 1)
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"save to model with valid_loss = {valid_loss:.3f}",
                  '    to best_' + dataset_name + '_saved_weights.pth')
            torch.save(model, 'best_' + dataset_name + '_saved_weights.pth')
        else:
            print(f"is not better than valid_loss =  {best_valid_loss:.3f}")

        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}| Valid Loss: {valid_loss:.3f} | '
            f'Valid Acc: {valid_acc * 100:.2f}')
        print(
            f'\tValid Recall: {valid_recall * 100:.2f}| '
            f'Valid Precision: {valid_precision * 100:.2f}| Valid f1: {valid_f1 * 100:.2f}| Valid Auc: '
            f'{valid_auc * 100:.2f}')
    print(total_valid_loss)
    print(total_valid_recall)
    print(total_valid_f1)
    print(total_valid_precision)
    print(total_valid_valid_auc)

print("-------------------------------------------------------------")
print(
    f'\t| Valid Loss: {np.mean(total_valid_loss):.3f} | '
    f'Valid Acc: {np.mean(total_valid_acc) * 100:.2f}')
print(
    f'\tValid Recall: {np.mean(total_valid_recall) * 100:.2f}| '
    f'Valid Precision: {np.mean(total_valid_precision) * 100:.2f}| Valid f1: {np.mean(total_valid_f1) * 100:.2f}|'
    f' Valid Auc: '
    f'{np.mean(total_valid_valid_auc) * 100:.2f}')
