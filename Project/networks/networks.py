import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torchvision.ops import Conv2dNormActivation

from networks.inceptionTime import Inception, InceptionBlock
from torchvision import models


functions = {
    # activation functions
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    # layers
    "identity": nn.Identity,
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "linear": nn.Linear,
    "lazylinear": nn.LazyLinear,
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

    "efficientnet_b2": models.efficientnet_b2()
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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # Compute attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention weights to LSTM outputs
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)

        return weighted_output, attention_weights

class RNN(nn.Module):
    def __init__(self, nn_config: dict):
        super().__init__()
        self.bn = []
        self.lstm = []
        self.output = []
        for layer_config in nn_config["bn_config"]:
            self.bn.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.bn = nn.Sequential(*self.bn)
        for layer_config in nn_config["lstm_config"]:
            self.lstm.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.lstm = nn.Sequential(*self.lstm)
        self.input_size=nn_config["lstm_config"][0]["kwargs"]["input_size"]
        self.hidden_size=nn_config["lstm_config"][0]["kwargs"]["hidden_size"]

        for layer_config in nn_config["output_config"]:
            self.output.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.output = nn.Sequential(*self.output)

    def forward(self, x):
        x = self.bn(x)
        x = x.reshape(x.size(0), -1, self.input_size)

        lstm_out, (h0, c0) = self.lstm(x)
        # print(lstm_out.shape)
        output = self.output(lstm_out)
        # print(output.shape)

        return output


class RnnFcn(nn.Module):
    def __init__(self, nn_config: dict):
        super().__init__()
        self.bn = []
        self.lstm = []
        self.fcn = []
        self.output = []
        for layer_config in nn_config["batchnorm_config"]:
            self.lstm.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.bn = nn.Sequential(*self.bn)
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
        x = self.bn(x)
        lstm_output, _ = self.lstm(input=x)

        fcn_output = self.fcn(x)
        lstm_output = torch.flatten(lstm_output, start_dim=1)
        fcn_output = torch.flatten(fcn_output, start_dim=1)

        concat_output = torch.cat((fcn_output, lstm_output), 1)

        output = self.output(concat_output)
        return output
class RnnFcn_other(nn.Module):
    def __init__(self, nn_config: dict):
        super().__init__()
        self.bn = []
        self.lstm = []
        self.fcn = []
        self.output = []
        for layer_config in nn_config["batchnorm_config"]:
            self.lstm.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.bn = nn.Sequential(*self.bn)
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
        x = self.bn(x)
        lstm_output, _ = self.lstm(input=x)

        fcn_output = self.fcn(lstm_output)

        fcn_output = torch.flatten(fcn_output, start_dim=1)

        #concat_output = torch.cat((fcn_output, lstm_output), 1)

        output = self.output(fcn_output)
        return output
class ResNet(nn.Module):
    def __init__(self, nn_config:list[dict], num_classes=1):
        super().__init__()
        self.layers = []
        for layer_config in nn_config:
            self.layers.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.layers = nn.Sequential(*self.layers)

        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 6)

    def forward(self, x):
        x = self.layers(x)
        x = self.model(x)

        return x

class EfficientNet(nn.Module):
    def __init__(self, nn_config: list[dict], num_classes=1):
        super().__init__()
        self.layers = []
        for layer_config in nn_config:
            self.layers.append(create_layer(layer_config["name"], **layer_config.get("kwargs", {})))
        self.layers = nn.Sequential(*self.layers)


        self.model = models.efficientnet_b3()
        self.model.features[0] = Conv2dNormActivation(
                1, 32, kernel_size=3, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.model(x)
        return x
