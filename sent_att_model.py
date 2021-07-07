import torch
import torch.nn as nn
import torch.nn.functional as F

from HANtest.utils import *


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True, num_layers=1)
        # self.fc = nn.Linear(2 * sent_hidden_size, 2)
        # self.act = nn.Sigmoid()
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input):
        f_output, h_output = self.gru(input)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        # output = self.fc(output)
        # output = self.act(output)
        return output, h_output


if __name__ == "__main__":
    abc = SentAttNet()
