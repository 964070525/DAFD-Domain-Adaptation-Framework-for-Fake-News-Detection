import torch
import torch.nn as nn

from HANtest.WordAttNet import WordAttNet
from HANtest.sent_att_model import SentAttNet
from HANtest.utils import mmd_rbf
from mmd import mmd_linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device("cpu")


class HierAttNet_finetune(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet_finetune, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        # self._init_hidden_state()
        # self.drop = nn.Dropout(0.15)

        self.fc = nn.Linear(128, 2)
        self.act = nn.Sigmoid()

    # def _init_hidden_state(self, last_batch_size=None):
    #     if last_batch_size:
    #         batch_size = last_batch_size
    #     else:
    #         batch_size = self.batch_size
    #     self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
    #     self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
    #
    #     self.word_hidden_state = self.word_hidden_state.to(device)
    #     self.sent_hidden_state = self.sent_hidden_state.to(device)

    def forward(self, input, input2=None):
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0))
            output_list.append(output)
        output_from_word = torch.cat(output_list, 0)
        # output_from_word = self.drop(output_from_word)
        output, self.sent_hidden_state = self.sent_att_net(output_from_word)
        # output = self.drop(output)
        output = self.fc(output)
        output = self.act(output)
        return output
