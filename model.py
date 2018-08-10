from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from rnet import RepresentNet, TDClass
from utils import norm_col_init, weights_init

class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space, pre_rnet='None'):
        super(A3Clstm, self).__init__()

        self.lstm_1 = nn.LSTMCell(1024, 512)
        self.lstm_2 = nn.LSTMCell(1024, 512)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm_1.bias_ih.data.fill_(0)
        self.lstm_2.bias_ih.data.fill_(0)
        self.lstm_1.bias_hh.data.fill_(0)
        self.lstm_2.bias_hh.data.fill_(0)
        if pre_rnet=='None':
            self.r_net = RepresentNet()
            # self.c_net = TDClass()
        else:
            self.r_net = torch.load("pre_model/r_net_{}.pkl".format(pre_rnet))
            # self.c_net = torch.load("pre_model/c_net_{}.pkl".format(pre_rnet))

        self.train()

    def forward(self, inputs):
        inputs, (hx1, cx1), (hx2, cx2) = inputs
        
        x = self.r_net(inputs)  # .detach()

        hx1, cx1 = self.lstm_1(x, (hx1, cx1))
        hx2, cx2 = self.lstm_2(x, (hx2, cx2))

        x1 = hx1
        x2 = hx2

        return self.critic_linear(x1), self.actor_linear(x2), (hx1, cx1), (hx2, cx2)
