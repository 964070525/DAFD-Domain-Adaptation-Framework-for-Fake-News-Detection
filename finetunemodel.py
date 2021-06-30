import torch
from torch import nn


class fine_model(nn.Module):

    # 定义所有层
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()

        # embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm 层
        self.lstm = nn.GRU(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        # 全连接层
        self.fc2 = nn.Linear(2 * hidden_dim, 2)

        # 激活函数
        self.act = nn.Sigmoid()

    def forward(self, text, text2, text_lengths, text2_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, hidden = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]
        # print(hidden.shape)
        # 连接最后的正向和反向隐状态
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # print(hidden.shape)
        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc2(hidden)
        # 激活
        outputs = self.act(dense_outputs)

        return outputs
