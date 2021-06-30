import random
import os
from torch import optim, nn
from dataset import *
from model import classifier
from train_model import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
SEED = 2019
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
dataset_name = "politifact"
dataset_name2 = "gossipcop"
# gossipcop politifact Constraint
source_data = get_Dataset(dataset_name)
target_data = get_Dataset(dataset_name2)
all_data = get_Dataset(dataset_name, dataset_name2)
build_vocab(all_data)
train_data, valid_data = source_data.split(split_ratio=0.75, random_state=random.seed(SEED))
train_data1, valid_data1 = target_data.split(split_ratio=0.05, random_state=random.seed(SEED))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置batch大小
BATCH_SIZE = 64

# 载入迭代器
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)
train_iterator1, valid_iterator1 = data.BucketIterator.splits(
    (train_data1, valid_data1),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)

size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 50
num_output_nodes = 2
num_layers = 2
bidirection = True
dropout = 0.2

# 实例化模型
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers,
                   bidirectional=True, dropout=dropout)
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
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
loss_func = nn.CrossEntropyLoss()

# 如果cuda可用
model = model.to(device)
criterion = loss_func.to(device)

N_EPOCHS = 30
best_valid_loss = float('inf')
print("train on", len(train_data), "    Valid on", len(valid_data))
lr_list = []
if train_iterator1 is not None:
    for epoch in range(N_EPOCHS):

        # 训练模型
        train_loss, loss1, mmd_loss, train_acc, train_recall, train_f1, train_precision, train_auc = train(model,
                                                                                                           train_iterator,
                                                                                                           train_iterator1,
                                                                                                           optimizer,
                                                                                                           criterion)

        # 评估模型
        valid_loss, valid_acc, valid_recall, valid_f1, valid_precision, valid_auc = evaluate(model, valid_iterator1,
                                                                                             criterion)
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        print(
            "epoch: ", epoch + 1)
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"save to model with _loss = {best_valid_loss:.3f}",
                  'to  ' + 'p_g_gru_05.pth')
            torch.save(model, 'p_g_gru_05.pth')
        else:
            print(f"is not better than valid_loss =  {best_valid_loss:.3f}")

        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}| Valid Loss: {valid_loss:.3f} | '
            f'Valid Acc: {valid_acc * 100:.2f}| Other Loss: {loss1:.3f}| mmd Loss: {mmd_loss:.3f}')
        print(
            f'\tValid Recall: {valid_recall * 100:.2f}| '
            f'Valid Precision: {valid_precision * 100:.2f}| Valid f1: {valid_f1 * 100:.2f}| Valid Auc: '
            f'{valid_auc * 100:.2f}')
