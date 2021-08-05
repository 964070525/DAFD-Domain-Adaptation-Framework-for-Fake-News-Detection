import os
import pickle

import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
import argparse

from HANtest.dataset import MyDataset
from HANtest.hierarchical_att_model import HierAttNet
from HANtest.train_model import train_without_mmd, evaluate
from HANtest.utils import *

total_valid_loss = []
total_valid_acc = []
total_valid_recall = []
total_valid_f1 = []
total_valid_precision = []
total_valid_valid_auc = []
dataset_name = "gossipcop"


# gossipcop politifact Constraint
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


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoches", type=int, default=20)
    parser.add_argument("--word_hidden_size", type=int, default=64)
    parser.add_argument("--sent_hidden_size", type=int, default=64)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="../data/" + dataset_name + "_content_no_ignore.tsv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="../glove.6B.100d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": False}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    all_data = MyDataset(opt.train_set, opt.word2vec_path)


    f = open(dataset_name + "training_set_10", "rb")
    training_set = pickle.load(f)
    f = open(dataset_name + "test_set_10", "rb")
    test_set = pickle.load(f)

    f.close()
    loss_func = nn.CrossEntropyLoss()
    criterion = loss_func.to(device)
    for i in range(0, 1):
        model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, 2,
                           opt.word2vec_path, all_data.max_length_sentences, all_data.max_length_word)
        model = model.to(device)
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        best_loss = 10000
        model.train()
        # training_set, test_set = change_dataset(i, training_set, test_set)
        training_generator = DataLoader(training_set, **training_params)
        test_generator = DataLoader(test_set, **test_params)
        print("train on", len(training_set), "    Valid on", len(test_set))
        for epoch in range(opt.num_epoches):
            model.train()
            train_loss, train_acc, train_recall, train_f1, train_precision, train_auc = train_without_mmd(
                epoch,
                model,
                training_generator,
                optimizer,
                criterion, True)
            if np.isnan(train_loss):
                print("nan")
                break
            model.eval()
            valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_auc = evaluate(model, test_generator,
                                                                                                 criterion, True)
            change_evaluating(i, valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_auc)
            scheduler.step()
            print(
                "epoch: ", epoch + 1)
            if valid_loss < best_loss:
                best_loss = valid_loss
                print(f"save to model with valid_loss = {valid_loss:.3f}",
                      '    to best_' + dataset_name + '_saved_weights.pth')
                # torch.save(model, 'best_' + dataset_name + '_saved_weights.pth')
            else:
                print(f"is not better than valid_loss =  {best_loss:.3f}")
            print(
                f'\tTrain Loss: {train_loss :.3f} | Train Acc: {train_acc * 100:.2f}| '
                f'Valid Loss: {valid_loss :.3f} | '
                f'Valid Acc: {valid_acc * 100:.2f}')
            print(
                f'\tValid Recall: {valid_recall * 100:.2f}| '
                f'Valid Precision: {valid_precision * 100:.2f}| Valid f1: {valid_f1 * 100:.2f}'
                f'| Valid Auc: {valid_auc * 100:.2f}')

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


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    SEED = 2019
    torch.manual_seed(SEED)
    opt = get_args()
    train(opt)
