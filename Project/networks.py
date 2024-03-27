from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from icecream import ic
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from inceptionTime import Inception, InceptionBlock
from signal_dataset import SignalDataset

DEVICE = 'cuda'
torch.manual_seed(22)


class MLP(nn.Module):
    def __init__(self, layers_config_list: list[dict]):
        super().__init__()
        self.layers = []
        for layers_config_dict in layers_config_list:
            for i, layers_config in layers_config_dict.items():
                for layer_config in layers_config:
                    self.layers.append(
                        eval('nn.' + layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class CNN(nn.Module):
    def __init__(self, blocks_config: list[dict]):
        super().__init__()

        self.layers = []

        for layers_config_dict in blocks_config:
            for i, layers_config in layers_config_dict.items():
                for layer_config in layers_config:
                    self.layers.append(
                        eval('nn.' + layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        self.layers = nn.ModuleList(self.layers)

        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        # print(x.shape)
        # x = torch.unsqueeze(x, 1)
        x = x.view(-1, 1, 1000)
        # for layer in self.conv_layers:
        #     x = F.relu(layer(x))
        # x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.AdaptiveAvgPool1d) or isinstance(layer, nn.Flatten):
                x = layer(x)
            else:
                x = F.relu(layer(x))
        x = self.layers[-1](x)

        return x


class CNNOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 200, padding=50)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.bn = nn.BatchNorm1d(991)
        self.fc1 = nn.Linear(901, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 9)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class InceptionTime(nn.Module):
    def __init__(self, blocks_config: list[dict]):
        super().__init__()
        self.layers = []
        for layers_config_dict in blocks_config:
            for i, layers_config in layers_config_dict.items():
                for layer_config in layers_config:
                    self.layers.append(
                        eval(layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        self.layers = nn.ModuleList(self.layers)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.lin = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = torch.reshape(input=x, shape=(-1, 1, 1000))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.lin(x)
        return x


class AttentionRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1000, hidden_size=512, num_layers=1, batch_first=True)
        self.bn = nn.BatchNorm1d(num_features=1)
        # self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=6, stride=2)
        self.attention = nn.Linear(512, 512)

        self.linear = nn.Linear(512, 9)

    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        out, (h0, c0) = self.rnn(out)
        # out = self.bn(out)
        # out = torch.permute(out, (0, 2, 1))
        # out = F.relu(self.conv(x))
        out = torch.flatten(10*out, start_dim=1)
        attention_weights = F.softmax(self.attention(out), dim=1)
        z = torch.sum(attention_weights * out, dim=1)

        z = self.linear(z)
        return z


class RNN(nn.Module):
    def __init__(self, blocks_config: list[dict]):
        super().__init__()
        self.layers = []
        for layers_config_dict in blocks_config:
            for i, layers_config in layers_config_dict.items():
                for layer_config in layers_config:
                    self.layers.append(
                        eval('nn.' + layer_config['name'])(*layer_config.get('args', []),
                                                           **layer_config.get('kwargs', {})))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        for layer in self.layers[:-1]:
            x, _ = layer(input=x)
        x = self.layers[-1](x)
        return x[:, -1, :]
