# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 问答模型的模块

if __name__ == '__main__':
	import sys
	sys.path.append('../')


import torch

from torch.nn import Module, LSTM, Linear, functional as F

from setting import *

class BaseLSTMEncoder(Module):
    def __init__(self):
        super(BaseLSTMEncoder, self).__init__()
        self.d_hidden = 128
        self.d_output = self.d_hidden
        self.is_bidirectional = True
        self.n_layers = 2
        if self.is_bidirectional:
            self.d_output = self.d_output // 2
        self.lstm = LSTM(input_size=self.d_hidden,
                         hidden_size=self.d_output,
                         num_layers=self.n_layers,
                         batch_first=True,
                         bidirectional=self.is_bidirectional)

    def forward(self, x):
        batch_size = x.size()[0]
        initial_states = (torch.autograd.Variable(torch.zeros(self.n_layers + int(self.is_bidirectional) * self.n_layers, batch_size, self.d_output)).to(DEVICE),
                          torch.autograd.Variable(torch.zeros(self.n_layers + int(self.is_bidirectional) * self.n_layers, batch_size, self.d_output)).to(DEVICE))
        hidden_output, final_states = self.lstm(x, initial_states)
        max_hidden_output = torch.max(hidden_output, dim=1)[0]
        return max_hidden_output, hidden_output


class BaseAttention(Module):
    def __init__(self):
        super(BaseAttention, self).__init__()
        self.d_hidden = 128
        self.linear = Linear(self.d_hidden, self.d_hidden)

    def forward(self, x, y):
        _x = self.linear(x)
        _y = torch.transpose(y, 1, 2)
        attention = torch.bmm(_x, _y)
        x_attention = torch.softmax(attention, dim=2)
        x_attention = torch.bmm(x_attention, y)
        y_attention = torch.softmax(attention, dim=1)
        y_attention = torch.bmm(torch.transpose(y_attention, 2, 1), x)
        return x_attention, y_attention, attention


