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

DEVICE = 'cuda'
torch.manual_seed(22)


class SineDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_count):
        self.freq = 10 * torch.rand(size=(data_count, 1), device=DEVICE) + 1

        random_uniform_shift = 10 * torch.rand(size=(data_count, 1), device=DEVICE)
        random_normal_shift = torch.normal(0, 1, size=(1, 1), device=DEVICE)

        self.points = torch.arange(0, 6, 1 / 40, device=DEVICE).repeat(data_count, 1) + random_uniform_shift
        self.phase = torch.normal(0, 5, size=(data_count, 1), device=DEVICE)
        self.amplitude = (self.points[0] - random_uniform_shift[0]) ** 2
        # print(self.amplitude)
        self.data_matrix = torch.sin(self.points * self.freq + self.phase)
        self.data_matrix_without_noise = torch.sin(self.points * self.freq + self.phase)
        for i in range(data_count):
            if random.random() < 0.3:
                self.amplitude = torch.flip(self.amplitude, dims=(-1,))
                random_noise = (self.amplitude + 1) * torch.normal(0, 1, size=(1, len(self.data_matrix[0])),
                                                                   device=DEVICE)
                self.data_matrix_without_noise[i] = self.amplitude * self.data_matrix[i]
                self.data_matrix[i] = self.amplitude * self.data_matrix[i] + random_noise
            elif 0.3 <= random.random() < 0.7:
                random_noise = (self.amplitude + 1) * torch.normal(0, 1, size=(1, len(self.data_matrix[0])),
                                                                   device=DEVICE)
                self.data_matrix_without_noise[i] = self.amplitude * self.data_matrix[i]
                self.data_matrix[i] = self.amplitude * self.data_matrix[i] + random_noise
            else:
                random_noise = torch.normal(0, 1, size=(1, len(self.data_matrix[0])), device=DEVICE)
                self.data_matrix_without_noise[i] = self.data_matrix[i]
                self.data_matrix[i] = self.data_matrix[i] + random_noise

    def __len__(self):
        return len(self.freq)

    def __num__(self):
        return

    def __getitem__(self, idx):
        return self.data_matrix[idx], self.freq[idx]

class ConvLayer
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

class RNN(nn.Module):
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


class NeuroNet:
    def __init__(self):
        self.model = self.build_model()
        self.model.to('cuda')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100)
        self.criterion = nn.MSELoss()
        self.history = []
        self.mse_avg = 0

    def build_model(self):
        return RNN()

    def train_model(self, training_data: Dataset, testing_data: Dataset):
        writer = SummaryWriter(comment="rnn_with_attention2")
        step = 0
        epoch_num = 100
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        test_dataloader = DataLoader(testing_data, batch_size=128, shuffle=True)

        for epoch in range(epoch_num):
            self.model.train()
            self.history = []
            for batch_id, (inputs, targets) in enumerate(
                    tqdm(train_dataloader, desc=f'epoch {epoch + 1}/ {epoch_num}')):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                writer.add_scalar('Training Loss', loss, global_step=step)
                step += 1

            self.model.eval()
            with torch.no_grad():
                for batch_id, (inputs, targets) in enumerate(test_dataloader):
                    outputs = self.model(inputs)
                    mse = np.mean((outputs.detach().cpu().numpy() - targets.detach().cpu().numpy()) ** 2)
                    self.history.append(mse)
                    self.mse_avg = sum(self.history) / len(self.history)
                    writer.add_scalar('validation loss', self.mse_avg, global_step=epoch)

            ic(self.scheduler.get_last_lr())
            last_lr = self.scheduler.get_last_lr()[0]
            writer.add_scalar('learning rate', last_lr, global_step=epoch)
            self.scheduler.step()
            print(self.mse_avg)
        writer.close()
        # torch.optim.lr_scheduler.print_lr()

    def predict(self, data: torch.Tensor):
        with torch.no_grad():
            points, freq = data
            return print(f'predicted freq: {self.model(points)}, real freq: {freq}')


dataset = SineDataset(128 * 128)
train_data, test_data = random_split(dataset, [100*128, 28*128])

neuro_net = NeuroNet()

neuro_net.train_model(train_data, test_data)
