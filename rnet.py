import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init
class RepresentNet(nn.Module):
    def __init__(self):
        super(RepresentNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        
    def forward(self, inputs):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        return x  # F.relu(self.l_2(output))

class TDClass(nn.Module):
    def __init__(self):
        super(TDClass, self).__init__()
        # self.l_1 = nn.Linear(512, 128)
        self.l_1 = nn.Linear(512, 6)
        
    def forward(self, input_1, input_2, p):
        input = input_1 * input_2
        output = F.leaky_relu(self.l_1(input), negative_slope=0.2)
#         output = F.leaky_relu(self.l_2(output), negative_slope=0.2)
#         if p:
#             print(output)
        output = F.softmax(output, dim=1)
        return output
