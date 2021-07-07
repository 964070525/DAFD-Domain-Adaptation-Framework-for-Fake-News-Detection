import pickle
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from HANtest.fintunemodel import HierAttNet_finetune
from HANtest.train_model import train_without_mmd, evaluate, train_with_mmd
from HANtest.utils import *
import warnings

SEED = 2019
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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


batch_size = 50
warnings.filterwarnings("ignore")
training_params = {"batch_size": batch_size,
                   "shuffle": True,
                   "drop_last": False}
test_params = {"batch_size": batch_size,
               "shuffle": False,
               "drop_last": False}
dataset_name = "gossipcop"
# gossipcop politifact Constraint
f = open(dataset_name + "training_set_15", "rb")
training_set = pickle.load(f)
f = open(dataset_name + "test_set_15", "rb")
test_set = pickle.load(f)
f.close()


criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
best_loss = 100
# best_Constraint_saved_weights
# model = torch.load("c_g_HAN_10.pth")
model = torch.load("c_g_HAN_15.pth")
model.eval()
loss_func = nn.CrossEntropyLoss()
criterion = loss_func.to(device)
training_generator = DataLoader(training_set, **training_params)
test_generator = DataLoader(test_set, **test_params)
# 取消autograd
with torch.no_grad():
    valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_auc = evaluate(model, test_generator,
                                                                                         criterion, True)
    print(
        f'\tValid Recall: {valid_recall * 100:.2f}| '
        f'Valid Precision: {valid_precision * 100:.2f}| Valid f1: {valid_f1 * 100:.2f}'
        f'| Valid Auc: {valid_auc * 100:.2f}')
model = torch.load("c_g_HAN_15.pth")
# print(model)
# model.fc = nn.Linear(100, 2, bias=True)
model = model.to(device)
print("train on ", len(training_set), "     valid on ", len(test_set))
for i in range(0, 1):
    my_model = HierAttNet_finetune(64, 64, batch_size, 2, "../glove.6B.100d.txt", 100, 150)
    pretrained_dict = model.state_dict()
    model_dict = my_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and k != 'fc.weight' and k != "fc.bias"}
    model_dict.update(pretrained_dict)
    my_model.load_state_dict(model_dict)
    # for p in my_model.parameters():
    #     p.requires_grad = False
    # for para in my_model.fc.parameters():
    #     para.requires_grad = True
    print(my_model)
    my_model = my_model.to(device)
    # training_set, test_set = change_dataset(i, training_set, test_set)
    # training_generator = DataLoader(training_set, **training_params)
    # test_generator = DataLoader(test_set, **test_params)
    optimizer = optim.Adam(my_model.parameters(), lr=0.0071)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    for epoch in range(30):
        if epoch == 5:
            for p in my_model.parameters():
                p.requires_grad = True
        my_model.train()
        train_loss, train_acc, train_recall, train_f1, train_precision, train_auc = train_without_mmd(
            my_model,
            training_generator,
            optimizer,
            criterion,
            True)

        my_model.eval()
        valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_auc = evaluate(my_model, test_generator,
                                                                                             criterion, True)
        change_evaluating(i, valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_auc)
        scheduler.step()
        print(
            "epoch: ", epoch + 1)

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
    print(total_valid_acc)
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
