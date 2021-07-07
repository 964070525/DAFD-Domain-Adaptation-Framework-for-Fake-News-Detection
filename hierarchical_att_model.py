import torch
import torch.nn as nn

from HANtest import mmd
from HANtest.WordAttNet import WordAttNet
from HANtest.sent_att_model import SentAttNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        # self.max_sent_length = max_sent_length
        # self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        # self.fc1 = nn.Linear(2 * sent_hidden_size, 256)
        self.fc = nn.Linear(2 * sent_hidden_size, 2)
        self.act = nn.Sigmoid()
        # self._init_hidden_state()

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

    def forward(self, input, input2=None, needoutput=False):

        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0))
            output_list.append(output)
        output_from_word = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output_from_word)
        # source_gru = torch.cat((self.sent_hidden_state[-2, :, :], self.sent_hidden_state[-1, :, :]), dim=1)
        output_source = self.fc(output)
        output_source = self.act(output_source)
        if input2 != None:
            output_list2 = []
            input2 = input2.permute(1, 0, 2)
            for i in input2:
                output2, self.word_hidden_state2 = self.word_att_net(i.permute(1, 0))
                output_list2.append(output2)
            output_from_word2 = torch.cat(output_list2, 0)
            output2, self.sent_hidden_state2 = self.sent_att_net(output_from_word2)
            # target_gru = torch.cat((self.sent_hidden_state[-2, :, :], self.sent_hidden_state[-1, :, :]), dim=1)
            mmd_loss = mmd.MMD_loss()  # kernel_type="linear"
            output_source2 = self.fc(output2)
            output_source2 = self.act(output_source2)
            loss = mmd_loss(output, output2)
            if needoutput:
                return output, output2, output_source, output_source2, loss
            return output_source, output_source2, loss
        if needoutput:
            return output, output_source
        return output_source
