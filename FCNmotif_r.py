# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
from torch.autograd import  Variable


import numpy as np
import sys


# define simple rnn
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_class, num_layers=1):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # pytorch rnn
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # fully connected layer
        self.fc1 = nn.Linear(hidden_size, 64)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, out_class)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inputs = inputs.view(len(inputs), 1, -1)
        output, _ = self.rnn(inputs)
        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.relu(output)
        output = self.drop(output)
        output = self.fc2(output)
        out_class = self.sigmoid(output)
        return out_class


# twolayers c_in=256 101
# (100,4,101) -> (100,64,87) -> (100,64,21) -> (100,64,17) -> (100,64,4)
# (100,1,101) -> (100,64,87) -> (100,64,21) -> (100,64,17) -> (100,64,4)
class FCN(nn.Module):
    # FPN for semantic segmentation

    def __init__(self, motiflen=15):
        super(FCN, self).__init__()  # 初始化
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=motiflen)  # conv4 and conv5 for conservation
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool5 = nn.MaxPool1d(kernel_size=4, stride=4)
        # classifier head
        c_in = 256
        hidden = 64
        c_out = 1
        self.rnn = SimpleRNN(c_in, hidden, c_out)
        # self.linear1 = nn.Linear(c_in, 64)
        # self.drop = nn.Dropout(p=0.5)
        # self.linear2 = nn.Linear(64, 1)
        # # general functions
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        # Initialize the new built layers
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data1, data2):
        # Construct a new computation graph at each froward
        b, _, _ = data1.size()
        # encode process
        out1 = self.conv1(data1)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip1 = out1
        out2 = self.conv4(data2)
        out2 = self.relu(out2)
        out2 = self.pool4(out2)
        out2 = self.dropout(out2)
        out2 = self.conv5(out2)
        out2 = self.relu(out2)
        out2 = self.pool5(out2)
        out2 = self.dropout(out2)
        skip2 = out2
        # classifier
        skip4 = skip1 + skip2  # add (100,64,4)
        # skip4 = torch.cat((skip1, skip2), 1)
        # classifier
        out3 = skip4.view(b, -1)
        out_class = self.rnn(out3)
        # out3 = self.linear1(out3)
        # out3 = self.relu(out3)
        # out3 = self.drop(out3)
        # out3 = self.linear2(out3)
        # out_class = self.sigmoid(out3)

        return out_class


# twolayers
class FCN1(nn.Module):
    # FPN for semantic segmentation

    def __init__(self, motiflen=15):
        super(FCN1, self).__init__()  # 初始化
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)  # 注意这里是一维卷积层，图像处理任务时才是二维卷积层
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=motiflen)  # 4、5、6用于处理进化信息
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool5 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv7 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=motiflen)  # 4、5、6用于处理进化信息
        self.pool7 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool8 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv10 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=motiflen)  # 4、5、6用于处理进化信息
        self.pool10 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv11 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool11 = nn.MaxPool1d(kernel_size=4, stride=4)
        # classifier head
        c_in = 256
        self.linear1 = nn.Linear(c_in, 64)
        self.drop = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        # Initialize the new built layers
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data1, data2, data3, data4):
        # Construct a new computation graph at each froward
        b, _, _ = data1.size()
        # encode process
        out1 = self.conv1(data1)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip1 = out1
        out2 = self.conv4(data2)
        out2 = self.relu(out2)
        out2 = self.pool4(out2)
        out2 = self.dropout(out2)
        out2 = self.conv5(out2)
        out2 = self.relu(out2)
        out2 = self.pool5(out2)
        out2 = self.dropout(out2)
        skip2 = out2
        out3 = self.conv7(data3)
        out3 = self.relu(out3)
        out3 = self.pool7(out3)
        out3 = self.dropout(out3)
        out3 = self.conv8(out3)
        out3 = self.relu(out3)
        out3 = self.pool8(out3)
        out3 = self.dropout(out3)
        skip3 = out3
        out4 = self.conv10(data4)
        out4 = self.relu(out4)
        out4 = self.pool10(out4)
        out4 = self.dropout(out4)
        out4 = self.conv11(out4)
        out4 = self.relu(out4)
        out4 = self.pool11(out4)
        out4 = self.dropout(out4)
        skip4 = out4
        # classifier
        skip = skip1 + skip2 + skip3 + skip4  # add
        # skip = torch.cat((skip1, skip2, skip3, skip4), 1)
        # classifier
        out = skip.view(b, -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.linear2(out)
        out_class = self.sigmoid(out)

        return out_class


class FCN2(nn.Module):
    # FPN for semantic segmentation

    def __init__(self, motiflen=15):
        super(FCN2, self).__init__()  # 初始化
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)  # 注意这里是一维卷积层，图像处理任务时才是二维卷积层
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=motiflen)  # 4、5、6用于处理进化信息
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool5 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv7 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=motiflen)  # 4、5、6用于处理进化信息
        self.pool7 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool8 = nn.MaxPool1d(kernel_size=4, stride=4)
        # classifier head
        c_in = 256
        self.linear1 = nn.Linear(c_in, 64)
        self.drop = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        # Initialize the new built layers
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data1, data2, data3):
        # Construct a new computation graph at each froward
        b, _, _ = data1.size()
        # encode process
        out1 = self.conv1(data1)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip1 = out1
        out2 = self.conv4(data2)
        out2 = self.relu(out2)
        out2 = self.pool4(out2)
        out2 = self.dropout(out2)
        out2 = self.conv5(out2)
        out2 = self.relu(out2)
        out2 = self.pool5(out2)
        out2 = self.dropout(out2)
        skip2 = out2
        out3 = self.conv7(data3)
        out3 = self.relu(out3)
        out3 = self.pool7(out3)
        out3 = self.dropout(out3)
        out3 = self.conv8(out3)
        out3 = self.relu(out3)
        out3 = self.pool8(out3)
        out3 = self.dropout(out3)
        skip3 = out3
        # classifier
        skip = skip1 + skip2 + skip3  # add
        # skip = torch.cat((skip1, skip2, skip3), 1)
        # classifier
        out = skip.view(b, -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.linear2(out)
        out_class = self.sigmoid(out)

        return out_class
