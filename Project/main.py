from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
from tqdm import trange
from signal_dataset import SignalDataset
import networks
import re
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

DEVICE = torch.device('cuda')
torch.manual_seed(21)


class NeuroNet:
    def __init__(self, control_center: Path):

        self.control_center = control_center

        self._load_yaml()
        self.model = self.build_model()
        self.model.to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.training_params["epoch_num"])
        self.criterion = nn.CrossEntropyLoss()
        self.history = []
        self.loss_avg = 0

    def _load_yaml(self) -> None:

        with self.control_center.open(mode="r") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.SafeLoader)

        self.training_params = data["training_params"]
        self.eval_params = data["eval_params"]
        self.lr = data["lr"]
        self.layers_config = data['layers']
        if self.control_center == Path('nn_yaml_configs/CNN.yaml'):
            self.conv_layers_config = data["conv_layers"]

    def build_model(self):
        if self.control_center == Path('nn_yaml_configs/MLP.yaml'):
            return networks.NN(self.layers_config)
        if self.control_center == Path('nn_yaml_configs/InceptionTime.yaml'):
            return networks.InceptionTime(self.layers_config)
        if self.control_center == Path('nn_yaml_configs/LSTM.yaml') or self.control_center == Path('yaml_configs/GRU.yaml'):
            return networks.RNN(self.layers_config)
        if self.control_center == Path('nn_yaml_configs/CNN.yaml'):
            # return networks.CNN(self.conv_layers_config, self.layers_config)
            return networks.CNNOld()

    def train_model(self, training_data: Dataset):
        step = 0
        train_dataloader = DataLoader(training_data, **self.training_params.get("dataloader_params", {}))

        epochs = trange(self.training_params["epoch_num"], ncols=100)  # , desc='Epoch #', leave=True)
        writer = SummaryWriter(comment=f"_{self.control_center.stem}_{self.training_params['epoch_num']}_{self.lr}")

        for epoch in epochs:
            self.model.train()
            for batch_id, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                # targets = F.one_hot(targets, num_classes = 9)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(outputs)
                # print(outputs.size())
                # print(targets)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                writer.add_scalar('Training Loss', loss, global_step=step)
                step += 1
                epochs.set_description(f"Epoch #{epoch + 1}")

                # epochs.refresh()

            last_lr = self.scheduler.get_last_lr()[0]
            writer.add_scalar('learning rate', last_lr, global_step=epoch)
            self.scheduler.step()
            writer.close()

    def test_model(self, testing_data: Dataset):
        self.model.eval()
        history = []
        test_dataloader = DataLoader(testing_data, **self.eval_params)
        with torch.no_grad():
            for batch_id, (inputs, targets) in enumerate(test_dataloader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                history.append(loss)
                self.loss_avg = sum(history) / len(history)
                # self.writer.add_scalar('validation loss', self.mse_avg, global_step=epoch)

            print(f"Test loss is {self.loss_avg}")

        # torch.optim.lr_scheduler.print_lr()

    def predict(self, data: torch.Tensor):
        with torch.no_grad():
            # print(type(data))
            input, label = data
            input = torch.reshape(input.to(DEVICE), (1, -1))
            output = self.model(input)
            output = torch.argmax(output)
            return print(f'predicted label: {output}, real label: {label}')

    def eval_model(self, testing_data: Dataset):

        test_dataloader = DataLoader(testing_data, **self.eval_params)

        self.model.eval()
        class_num = 9
        cm = np.zeros((class_num, class_num))

        with torch.no_grad():
            for (inputs, targets) in test_dataloader:
                inputs = inputs.to(DEVICE)
                # print(f"inpots are {inputs}")
                outputs = self.model(inputs)
                # print(f"outputs are {outputs}")
                predictions = torch.argmax(outputs, dim=1)
                # print(f"prediction is {predictions}")
                predictions = predictions.cpu()
                predictions = np.array(predictions)
                cm_p = confusion_matrix(targets, predictions, labels=np.arange(class_num))
                cm = cm + cm_p



        cm_collumn_sum = cm.sum(axis=0)

        plt.figure(figsize=(10, 10))
        plt.imshow(cm, cmap='Greens')
        for i in range(class_num):
            for j in range(class_num):
                plt.text(j, i, int(cm[i, j]), ha="center", va="bottom", color='gray')
                plt.text(j, i, j, ha="center", va="top", color='gray')

        # plt.matshow(cm)
        plt.show()

        correct_pred = 0
        total_pred = 0
        for i in range((len(cm_collumn_sum))):
            for j in range((len(cm_collumn_sum))):
                if i == j:
                    correct_pred = correct_pred + cm[i][j]
                    total_pred = total_pred + cm[i][j]
                else:
                    total_pred = total_pred + cm[i][j]
        accuracy = correct_pred / total_pred
        print(f'\n accuracy: {accuracy}')


signal_data_dir = "/home/petr/Documents/Motor_project/AE_PETR_motor/"
sr = 1562500

bin_setup = [{"label": i.stem, "interval": [0, 15 * sr], "bin_path": list(i.glob('*.bin'))[0]} for i in
             Path(signal_data_dir).glob('WUP*') if re.search(r'\d$', i.stem)]


sd = SignalDataset(step=1000, window_size=1000, bin_setup=bin_setup, device="cpu", source_dtype="float32")

train_data, test_data = random_split(sd, [0.8, 0.2])
# print(train_data[0])
neuro_net = NeuroNet(Path('nn_yaml_configs/InceptionTime.yaml'))

neuro_net.train_model(train_data)

neuro_net.eval_model(test_data)
