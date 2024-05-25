import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from inceptionTime import Inception, InceptionBlock

functions = {
    # activation functions
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    # layers
    "identity": nn.Identity,
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "linear": nn.Linear,
    "lstm": nn.LSTM,
    # inception
    "inceptionblock": InceptionBlock,
    "inception": Inception,
    # others
    "batchnorm1d": nn.BatchNorm1d,
    "batchnorm2d": nn.BatchNorm2d,
    "adaptiveavgpool1d": nn.AdaptiveAvgPool1d,
    "adaptiveavgpool2d": nn.AdaptiveAvgPool2d,
    "avgpool1d": nn.AvgPool1d,
    "maxpool1d": nn.MaxPool1d,
    "maxpool2d": nn.MaxPool2d,
    "flatten": nn.Flatten,
    "unflatten": nn.Unflatten,
    "dropout": nn.Dropout,
    "spectrogram": T.Spectrogram,
}


def create_layer(layer_name, **layer_kwargs):
    return functions[layer_name.lower()](**layer_kwargs)


class MLP(nn.Module):
    def __init__(self, nn_config: list[dict]):
        super().__init__()
        self.layers = []
        for layer_config in nn_config:
            self.layers.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class CNN(nn.Module):
    def __init__(self, nn_config: list[dict]):
        super().__init__()

        self.layers = []

        for layer_config in nn_config:
            self.layers.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):

        x = self.layers(x)

        return x


class CNNOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, 200, padding=50)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.bn = nn.BatchNorm1d(14352)
        self.fc1 = nn.Linear(14352, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 9)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class InceptionTime(nn.Module):
    def __init__(self, nn_config: list[dict]):
        super().__init__()

        self.layers = []
        for layer_config in nn_config:
            self.layers.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
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
        out, (h0, c0) = self.rnn(x)
        # out = self.bn(out)
        # out = torch.permute(out, (0, 2, 1))
        # out = F.relu(self.conv(x))
        out = torch.flatten(10 * out, start_dim=1)
        attention_weights = F.softmax(self.attention(out), dim=1)
        z = torch.sum(attention_weights * out, dim=1)

        z = self.linear(z)
        return z


class RNN(nn.Module):
    def __init__(self, nn_config: dict, attention: False):
        super().__init__()
        self.attention = attention
        self.lstm = []
        self.output = []
        for layer_config in nn_config["lstm_config"]:
            self.lstm.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.lstm = nn.Sequential(*self.lstm)

        for layer_config in nn_config["output_config"]:
            self.output.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.output = nn.Sequential(*self.output)

    def forward(self, x):
        if self.attention:
            y = self.att(x)
        lstm_out, (h0, c0) = self.lstm(x)
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return x


class RnnFcn(nn.Module):
    def __init__(self, nn_config: dict):
        super().__init__()
        self.lstm = []
        self.fcn = []
        self.output = []
        for layer_config in nn_config["lstm_config"]:
            self.lstm.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.lstm = nn.Sequential(*self.lstm)

        for layer_config in nn_config["fcn_config"]:
            self.fcn.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.fcn = nn.Sequential(*self.fcn)

        for layer_config in nn_config["output_config"]:
            self.output.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.output = nn.Sequential(*self.output)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        lstm_output, _ = self.lstm(input=x)

        fcn_output = self.fcn(x)

        lstm_output = torch.flatten(lstm_output, start_dim=1)
        fcn_output = torch.flatten(fcn_output, start_dim=1)
        concat_output = torch.cat((fcn_output, lstm_output), 1)

        output = self.output(concat_output)
        return output
