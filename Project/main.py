import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import seaborn as sns
import networks
from signal_dataset import SignalDataset

DEVICE = torch.device('cuda')


# torch.manual_seed(21)


class NeuroNet:
    def __init__(self, control_center: Path):

        self.control_center = control_center

        self._load_yaml()
        self.model = self._build_model()
        self.model.to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model_config["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.model_config["training_params"]["epoch_num"])

        self.criterion = nn.CrossEntropyLoss()
        self.history = []
        self.loss_avg = 0

    def _load_yaml(self) -> None:
        #  TODO: repair yaml_config loading
        with self.control_center.open(mode="r") as yaml_file:
            self.model_config = yaml.load(yaml_file, Loader=yaml.SafeLoader)
        self.layers_configs = []
        for name, layer_config in self.model_config["model"]["kwargs"].items():
            self.layers_configs.append({name: layer_config})
        print(self.layers_configs)
        # self.layers_config = data['layers']
        # if self.control_center == Path('nn_yaml_configs/CNN.yaml'):
        #     self.conv_layers_config = data["conv_layers"]

    def _build_model(self):
        # TODO: make in_channels as parameter
        # TODO: eval (from networks import network type)
        match self.model_config["model"]["class"]:
            case "MLP":
                return networks.MLP(self.layers_configs)
            case "Inception time" | "Inception" | "Inception_time":
                return networks.InceptionTime(self.layers_configs)
            case "LSTM" | "GRU":
                # return networks.RNN(self.layers_configs)
                return networks.RNN(self.layers_configs)
            case "CNN":
                return networks.CNNOld()

    def train_model(self, training_data: Dataset, testing_data: Dataset):

        train_dataloader = DataLoader(training_data, **self.model_config["training_params"].get("dataloader_params", {}))

        epochs = trange(self.model_config["training_params"]["epoch_num"], ncols=100)  # , desc='Epoch #', leave=True)
        writer = SummaryWriter(comment=f"_{self.control_center.stem}_{self.model_config['eval_params']['batch_size']}_"
                                       f"{self.model_config['training_params']['epoch_num']}_{self.model_config['lr']}")

        total_batch_id = 1  # TODO: total batch count and add validation metrics to tensorboard
        for epoch in epochs:

            for (inputs, targets) in train_dataloader:
                self.model.train()
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                # targets = F.one_hot(targets, num_classes = 9)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(outputs)
                # print(outputs.size())
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                writer.add_scalar('Training Loss', loss, global_step=total_batch_id)
                total_batch_id += 1
                if total_batch_id % 100 == 0:
                    self.test_model(testing_data, writer, total_batch_id)
                epochs.set_description(f"Epoch #{epoch + 1}")
            last_lr = self.scheduler.get_last_lr()[0]
            writer.add_scalar('learning rate', last_lr, global_step=epoch)
            self.scheduler.step()  # TODO: after every epoch reset the scheduler

            # epochs.refresh()
            #
            # self.eval_model(testing_data, writer, )
            writer.close()

    def test_model(self, testing_data: Dataset, writer, count):
        # TODO
        self.model.eval()
        y_pred = []  # save prediction
        y_true = []  # save ground truth
        class_num = 9

        cm = np.zeros((class_num, class_num))
        test_dataloader = DataLoader(testing_data, **self.model_config["eval_params"])
        with torch.no_grad():
            for (inputs, targets) in test_dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.history.append(loss)
                self.loss_avg = sum(self.history) / len(self.history)
                writer.add_scalar('validation loss', self.loss_avg, global_step=count)

                # CONFUSION MATRIX
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(predictions)  # save prediction
                targets = targets.data.cpu().numpy()
                y_true.extend(targets)

            cm = confusion_matrix(y_true, y_pred, labels=np.arange(class_num))

            df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in range(class_num)],
                                 columns=[i for i in range(class_num)])
            plt.figure(figsize=(12, 7))
            accuracy = accuracy_score(y_true, y_pred)
            writer.add_figure(tag="Confusion matrix", figure=sns.heatmap(df_cm, annot=True, fmt=".1f").get_figure(),
                              global_step=count)
            writer.add_scalar('Accuracy', accuracy, global_step=count)

        # # torch.optim.lr_scheduler.print_lr()

    def predict(self, data: torch.Tensor):
        with torch.no_grad():
            # print(type(data))
            input, label = data
            input = torch.reshape(input.to(DEVICE), (1, -1))
            output = self.model(input)
            output = torch.argmax(output)
            return print(f'predicted label: {output}, real label: {label}')

    def eval_model(self, testing_data: Dataset):

        test_dataloader = DataLoader(testing_data, **self.model_config["eval_params"])

        self.model.eval()
        class_num = 9

        y_pred = []  # save predction
        y_true = []  # save ground truth
        with torch.no_grad():
            for (inputs, targets) in test_dataloader:
                inputs = inputs.to(DEVICE)
                # print(f"inpots are {inputs}")
                outputs = self.model(inputs)
                # print(f"outputs are {outputs}")
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(predictions)  # save prediction
                targets = targets.data.cpu().numpy()
                y_true.extend(targets)

            cm = confusion_matrix(y_true, y_pred, labels=np.arange(class_num))

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
        for i in range((len(cm.sum(axis=0)))):
            for j in range((len(cm.sum(axis=0)))):
                if i == j:
                    correct_pred = correct_pred + cm[i][j]
                    total_pred = total_pred + cm[i][j]
                else:
                    total_pred = total_pred + cm[i][j]
        accuracy = correct_pred / total_pred
        print(f'\n accuracy: {accuracy}')


signal_data_dir = "/home/petr/Documents/Motor_project/AE_PETR_motor/"
sr = 1562500

glob = Path(signal_data_dir).glob('WUP*')
bin_setup = [{"label": i.stem, "interval": [0, 15 * sr], "bin_path": list(i.glob('*.bin'))[0]} for i in
             glob if re.search(r'\d$', i.stem)]

sd = SignalDataset(step=1000, window_size=1000, bin_setup=bin_setup, device="cpu", source_dtype="float32")

train_data, test_data = random_split(sd, [0.8, 0.2])
# print(train_data[0])
neuro_net = NeuroNet(Path('nn_yaml_configs/InceptionTime.yaml'))

neuro_net.train_model(train_data, test_data)
print(sd.label_dict)
neuro_net.eval_model(test_data)

# TODO: abc.ABC template on material model for prediction
# TODO: learn pycharm keybindings, etc.  ctrl, shift, alt, arrows; ctrl+alt+m/v
