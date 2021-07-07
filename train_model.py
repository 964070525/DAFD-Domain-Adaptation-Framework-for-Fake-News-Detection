import copy

import torch
import time

from HANtest.adversarial import ADT, FGM
from HANtest.utils import get_evaluation
from util import get_evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_without_mmd(epoch, model, training_generator, optimizer, criterion, need_init):
    # 初始化
    train_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_auc = 0
    # pgd = PGD(model)
    # K = 1
    # fgm = FGM(model)
    sample_total = []
    label_total = []
    for itr, (feature, label) in enumerate(training_generator):
        feature = feature.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # if need_init:
        #     model._init_hidden_state()
        sence, predictions = model(feature, None, True)
        xxx = copy.copy(sence).detach().cpu().numpy()
        yyy = copy.copy(label).detach().cpu().numpy()
        for i in xxx:
            sample_total.append(" ".join(str(x) for x in i))
        for i in yyy:
            label_total.append(i)
        loss = criterion(predictions + 1e-8, label)
        loss.backward()
        # pgd.backup_grad()
        # for t in range(K):
        #     pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
        #     if t != K - 1:
        #         model.zero_grad()
        #     else:
        #         pgd.restore_grad()
        #     predictions1 = model(feature, None)
        #     loss_adv = criterion(predictions1, label)
        #     loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        # pgd.restore()  # 恢复embedding参数
        optimizer.step()
        # 损失和精度
        train_loss += loss.item()

        # 计算二分类精度
        acc, val_precision, val_recall, val_f1, val_auc = get_evaluate(predictions, label)
        epoch_acc += acc.item()
        epoch_recall += val_recall.item()
        epoch_f1 += val_f1.item()
        epoch_precision += val_precision.item()
        epoch_auc += val_auc

    with open('txtfile/sence' + str(epoch + 1) + '.txt', 'w') as f:
        for i in sample_total:
            f.write(i)
            f.write('\r\n')

    with open('txtfile/label' + str(epoch + 1) + '.txt', 'w') as f:
        for i in label_total:
            f.write(str(i))
            f.write('\r\n')

    return train_loss / len(training_generator), epoch_acc / len(training_generator), epoch_recall / len(
        training_generator), epoch_f1 / len(
        training_generator), epoch_precision / len(training_generator), epoch_auc / len(training_generator)


def train_with_mmd(epoch, model, training_generator, training_generator2, optimizer, criterion):
    # 初始化
    train_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_auc = 0
    total_mmd = 0
    iter2 = iter(training_generator2)
    sample_total = []
    label_total = []
    flag = True
    for itr, (feature, label) in enumerate(training_generator):
        feature = feature.to(device)
        label = label.to(device)
        (feature2, label2) = iter2.__next__()
        feature2 = feature2.to(device)
        label2 = label2.to(device)
        optimizer.zero_grad()
        # model._init_hidden_state()
        sence, sence2, predictions, predictions2, mmd_loss = model(feature, feature2, True)
        xxx1 = copy.copy(sence).detach().cpu().numpy()
        yyy1 = copy.copy(label).detach().cpu().numpy()
        xxx2 = copy.copy(sence2).detach().cpu().numpy()
        yyy2 = copy.copy(label2).detach().cpu().numpy()
        for i in xxx1:
            sample_total.append(" ".join(str(x) for x in i))
        for i in yyy1:
            label_total.append(i)
        if flag:
            for i in xxx2:
                sample_total.append(" ".join(str(x) for x in i))
            for i in yyy2:
                label_total.append(i + 2)

        loss = criterion(predictions + 1e-8, label)

        total_loss = 1 * loss + 0.2 * mmd_loss
        total_mmd += mmd_loss.item()
        total_loss.backward()
        optimizer.step()
        # 损失和精度
        train_loss += loss.item()

        # 计算二分类精度
        acc, val_precision, val_recall, val_f1, val_auc = get_evaluate(predictions, label)
        epoch_acc += acc.item()
        epoch_recall += val_recall.item()
        epoch_f1 += val_f1.item()
        epoch_precision += val_precision.item()
        epoch_auc += val_auc
        if (itr + 1) % len(training_generator2) == 0:
            iter2 = iter(training_generator2)
            flag = False
    with open('do_txtfile/sence' + str(epoch + 1) + '.txt', 'w') as f:
        for i in sample_total:
            f.write(i)
            f.write('\r\n')

    with open('do_txtfile/label' + str(epoch + 1) + '.txt', 'w') as f:
        for i in label_total:
            f.write(str(i))
            f.write('\r\n')
    return train_loss / len(training_generator), total_mmd / len(training_generator), epoch_acc / len(
        training_generator), epoch_recall / len(
        training_generator), epoch_f1 / len(
        training_generator), epoch_precision / len(training_generator), epoch_auc / len(training_generator)


def train_with_mmd_only_one_class(model, training_generator, training_generator2, optimizer, criterion):
    # 初始化
    train_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_auc = 0
    total_mmd = 0
    iter2 = iter(training_generator2)
    for itr, (feature, label) in enumerate(training_generator):
        feature = feature.to(device)
        label = label.to(device)
        (feature2, label2) = iter2.__next__()
        feature2 = feature2.to(device)
        label2 = label2.to(device)

        optimizer.zero_grad()
        model._init_hidden_state()
        predictions, predictions2, mmd_loss = model(feature, feature2)
        # loss = criterion(predictions + 1e-8, label)
        loss2 = criterion(predictions2 + 1e-8, label2)
        total_loss = loss2 + 0.25 * mmd_loss
        total_mmd += mmd_loss.item()
        total_loss.backward()
        optimizer.step()
        # 损失和精度
        train_loss += total_loss.item()

        # 计算二分类精度
        acc, val_precision, val_recall, val_f1, val_auc = get_evaluate(predictions, label)
        epoch_acc += acc.item()
        epoch_recall += val_recall.item()
        epoch_f1 += val_f1.item()
        epoch_precision += val_precision.item()
        epoch_auc += val_auc
        if (itr + 1) % len(training_generator2) == 0:
            iter2 = iter(training_generator2)

    return train_loss / len(training_generator), total_mmd * 0.2 / len(training_generator), epoch_acc / len(
        training_generator), epoch_recall / len(
        training_generator), epoch_f1 / len(
        training_generator), epoch_precision / len(training_generator), epoch_auc / len(training_generator)


def evaluate(model, test_generator, criterion, need_init):
    # 初始化
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_auc = 0

    # 停用dropout层
    model.eval()

    # 取消autograd
    with torch.no_grad():
        for te_feature, te_label in test_generator:
            num_sample = len(te_label)

            te_feature = te_feature.to(device)
            te_label = te_label.to(device)
            with torch.no_grad():
                # if need_init:
                #     model._init_hidden_state(num_sample)
                te_predictions = model(te_feature, None)
            te_loss = criterion(te_predictions + 1e-8, te_label)
            epoch_loss += te_loss.item()
            acc, val_precision, val_recall, val_f1, val_auc = get_evaluate(te_predictions, te_label)
            epoch_acc += acc.item()
            epoch_recall += val_recall.item()
            epoch_f1 += val_f1.item()
            epoch_precision += val_precision.item()
            epoch_auc += val_auc

    return epoch_loss / len(test_generator), epoch_acc / len(test_generator), epoch_recall / len(
        test_generator), epoch_f1 / len(
        test_generator), epoch_precision / len(test_generator), epoch_auc / len(test_generator)
