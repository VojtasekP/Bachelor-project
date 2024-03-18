
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

class NN(nn.Module):
    def __init__(self, layers_config: list[dict]):
        super().__init__()
        self.layers = []
        for layer_config in layers_config:
            self.layers.append(
                eval('nn.' + layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class CNN(nn.Module):
    def __init__(self, conv_layers_config: list[dict], layers_config: list[dict]):  # todo: repair
        super().__init__()
        self.conv_layers = []
        self.layers = []
        for layer_config in conv_layers_config:
            self.conv_layers.append(
                eval('nn.' + layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        for layer_config in layers_config:
            self.layers.append(
                eval('nn.' + layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        self.layers = nn.ModuleList(self.layers)
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        for layer in self.conv_layers:
            x = F.relu(layer(x))

        x = torch.flatten(x, -1)

        for layer in self.fullycon_layers:
            x = F.relu(layer(x))
        return x


class InceptionTime(nn.Module):
    def __init__(self, layers_config: list[dict]):
        super().__init__()
        self.layers = []
        for layer_config in layers_config:
            self.layers.append(
                eval(layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        self.inception_layers = nn.ModuleList(self.layers)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.lin = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = torch.reshape(input=x, shape=(-1, 1, 240))
        for layer in self.inception_layers:
            x = F.relu(layer(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.lin(x))
        return x

class attention_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=240, hidden_size=64, num_layers=1, batch_first = True)
        self.bn = nn.BatchNorm1d(num_features=1)
        # self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=6, stride=2)
        self.attention = nn.Linear(64,64)

        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        out, (h0, c0) = self.rnn(out)
        # out = self.bn(out)
        # out = torch.permute(out, (0, 2, 1))
        # out = F.relu(self.conv(x))
        attention_weights = F.softmax(self.attention(out), dim=-1)
        repre = torch.sum(attention_weights * out, dim=1)
        out = torch.squeeze(repre, 1)
        out = F.relu(self.linear(out))
        return out

class RNN(nn.Module):
    def __init__(self, layers_config: list[dict]):
        super().__init__()
        self.layers = []
        for layer_config in layers_config:
            self.layers.append(
                eval('nn.' + layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        for layer in self.layers[:-1]:
            x, _ = layer(input=x)
        x = self.layers[-1](x)
        return x[:, -1, :]