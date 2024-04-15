import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import seaborn as sns
import networks
from signal_dataset import SignalDataset

DEVICE = torch.device('cuda')


# torch.manual_seed(21)
# TODO: SignalModel, Neuralnet/Skitlearn model inherit from SignalModel,train functioning same for all, predict
# compare with randomnumbergenerator model

class SignalModel:
    pass


class NeuroNet:
    def __init__(self, control_center: Path):

        self._load_yaml(control_center)
        self.model = self._build_model()
        self.model.to(DEVICE)

        self.pretrained = False
        self.criterion = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(
            comment=f"_{control_center.stem}_{self.model_config['eval_params']['batch_size']}")
        self.loss_avg = 0
        self.val_loss = []
        self.train_loss = []
        self.total_batch_id = 1
        self.epoch_trained = 0

    def _load_yaml(self, control_center) -> None:
        with control_center.open(mode="r") as yaml_file:
            self.config = yaml.load(yaml_file, Loader=yaml.SafeLoader)
            self.model_config = self.config["model"]

        if isinstance(self.model_config["kwargs"]["layers"], list):
            self.layers_configs = []
            for layer_config in self.model_config["kwargs"]["layers"]:
                self.layers_configs.append(layer_config)
        else:
            self.layers_configs = {}
            for name, kwargs in self.model_config["kwargs"]["layers"].items():
                self.layers_configs[name] = kwargs

    def _build_model(self):
        # TODO: make in_channels as parameter
        # TODO: eval (from networks import network type)
        match self.model_config["class"]:
            case "MLP":
                return networks.MLP(self.layers_configs)
            case "Inception time" | "Inception" | "Inception_time":
                return networks.InceptionTime(self.layers_configs)
            case "LSTM" | "GRU":
                # return networks.RNN(self.layers_configs)
                return networks.RNN(self.layers_configs, attention=self.model_config["attention"])
            case "CNN":
                return networks.CNNOld()
            case "LSTM-FCN":
                return networks.RnnFcn(self.layers_configs)

    def train_model(self, training_data: Dataset, testing_data: Dataset):

        self._load_yaml()

        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.config["training_params"][
                                                                   "epoch_num"])

        train_dataloader = DataLoader(training_data,
                                      **self.config["training_params"].get("dataloader_params", {}))
        test_dataloader = DataLoader(testing_data, **self.config["eval_params"])
        epochs = trange(self.config["training_params"]["epoch_num"], ncols=100)  # , desc='Epoch #', leave=True)

        for epoch in epochs:

            for (inputs, targets) in train_dataloader:
                self.model.train()
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                # targets = F.one_hot(targets, num_classes = 9)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(outputs)
                # print(outputs.size())
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                self.train_loss.append(loss)
                self.writer.add_scalar('Training Loss', loss, global_step=self.total_batch_id)
                self.total_batch_id += 1
                if self.total_batch_id % 50 == 0:
                    self.validation_loss(test_dataloader)
                    if self.total_batch_id % 200 == 0:
                        self.create_confusion_matrix(test_dataloader)
                epochs.set_description(f"Epoch #{self.epoch_trained + 1}")
            last_lr = scheduler.get_last_lr()[0]
            self.writer.add_scalar('learning rate', last_lr, global_step=self.epoch_trained)
            self.epoch_trained += 1
            scheduler.step()

            # epochs.refresh()
            #
            # self.eval_model(testing_data, writer, )

            self.pretrained = True

    #  TODO: acc on train_dataloader

    def create_confusion_matrix(self, test_dataloader: DataLoader):
        # TODO
        self.model.eval()
        y_pred = []  # save prediction
        y_true = []  # save ground truth
        class_num = 9
        cm = np.zeros((class_num, class_num))

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_dataloader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)

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
            self.writer.add_figure(tag="Confusion matrix",
                                   figure=sns.heatmap(df_cm, annot=True, fmt=".1f").get_figure(),
                                   global_step=self.total_batch_id)
            self.writer.add_scalar('Accuracy', accuracy, global_step=self.total_batch_id)

        # # torch.optim.lr_scheduler.print_lr()

    def validation_loss(self, test_dataloader: DataLoader):
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_dataloader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.val_loss.append(loss)
        self.loss_avg = sum(self.val_loss) / len(self.val_loss)
        self.writer.add_scalar('validation loss', self.loss_avg, global_step=self.total_batch_id)

    def classification_report(self, testing_data: Dataset):

        test_dataloader = DataLoader(testing_data, **self.config["eval_params"])

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
                plt.text(j, i, cm[i, j], ha="center", va="bottom", color='gray')
                plt.text(j, i, str(j), ha="center", va="top", color='gray')
        print(classification_report(y_true, y_pred))

    def predict(self, data: torch.Tensor):
        with torch.no_grad():
            # print(type(data))
            input, label = data
            input = torch.reshape(input.to(DEVICE), (1, -1))
            output = self.model(input)
            output = torch.argmax(output)
            return print(f'predicted label: {output}, real label: {label}')

    def close_writer(self):
        self.writer.close()


def main():
    signal_data_dir = "/home/petr/Documents/Motor_project/AE_PETR_motor/"
    sr = 1562500

    glob = Path(signal_data_dir).glob('WUP*')
    bin_setup = [{"label": i.stem, "interval": [0, 15 * sr], "bin_path": list(i.glob('*.bin'))[0]} for i in
                 Path(signal_data_dir).glob('WUP*') if re.search(r'\d$', i.stem)]

    sd = SignalDataset(step=1000, window_size=1000, bin_setup=bin_setup, device="cpu", source_dtype="float32")

    train_data, test_data = random_split(sd, [0.8, 0.2])
    # print(train_data[0])
    neuro_net = NeuroNet(Path('nn_yaml_configs/LSTM-FCN.yaml'))

    neuro_net.train_model(train_data, test_data)
    print(sd.label_dict)
    neuro_net.classification_report(test_data)
    neuro_net.close_writer()


if __name__ == '__main__':
    main()
# TODO: abc.ABC template on material model for prediction
# TODO: learn pycharm keybindings, etc.  ctrl, shift, alt, arrows; ctrl+alt+m/v
