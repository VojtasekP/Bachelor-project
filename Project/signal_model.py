
import abc
import re
from abc import ABC
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import yaml
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import networks
from signal_dataset import SignalDataset
import tsaug
DEVICE = "cuda"


class SignalModel:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._load_config()
        self._model = self.init_model()
        self._transform = self.init_transform()
        self._load_config()

    @abc.abstractmethod
    def _load_config(self):
        pass

    @abc.abstractmethod
    def init_model(self):
        pass

    @abc.abstractmethod  # raw outputs, targets
    def evaluate(self, dataset: Dataset) -> (np.ndarray, np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the label of one or multiple signals.
        """
        pass

    @abc.abstractmethod
    def save(self, path: Path):
        pass

    def init_transform(self) -> Callable:

        def transform(x):
            if isinstance(x, np.ndarray):
                x = np.swapaxes(x, 1, 2)

                x_aug = tsaug.AddNoise(scale=0.1).augment(x)
                return np.swapaxes(x_aug, 1, 2)
            else:
                print(x, x.shape, type(x))
        return transform

    # def _evaluate(self, x: np.ndarray) -> np.ndarray:
    #     x = self._transform(x)
    #     return self.inference(x)

    def plot_confusion_matrix(self, y_pred: np.ndarray, y_true: np.ndarray):
        class_num = 9  # in future will not be hard coded
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(class_num))
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, cmap='Greens')
        for i in range(class_num):
            for j in range(class_num):
                plt.text(j, i, cm[i, j], ha="center", va="bottom", color='gray')
                plt.text(j, i, str(j), ha="center", va="top", color='gray')

    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    def precision_and_recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> (float, float):
        cr = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        return cr["macro avg"]["precision"], cr["macro avg"]["recall"]

    def classification_report(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        return classification_report(y_true, y_pred, zero_division=0, output_dict=True)

class SklearnModel(SignalModel, ABC):

    def __init__(self, config_path: Path):
        super().__init__(config_path)

    def _load_config(self):
        with self.config_path.open(mode="r") as yaml_file:
            self.config = yaml.load(yaml_file, Loader=yaml.SafeLoader)
            self.model_config = self.config["model"]

    def init_model(self):
        match self.model_config["type"]:
            case "DummyClassifier":
                return DummyClassifier()
            case "RandomForestClassifier":
                return RandomForestClassifier()

    def train(self, train_dataset: Dataset, train_idx, test_idx) -> None:
        x_train = np.asarray([data[0] for data in train_dataset])
        x_train = self._transform(x_train)
        y_train = np.asarray([data[1] for data in train_dataset])
        self._model.fit(x_train, y_train)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)


class NeuroNet(SignalModel, ABC):

    def __init__(self, config_path: Path, state_dict: dict = None,
                 tensorboard: bool = False, validation_loss_check: bool = False,
                 fold: int = 1, k_folds: int = 1):
        super().__init__(config_path)
        self.tensorboard = tensorboard
        self.validation_loss_check = validation_loss_check
        # in case of CV, changing tensorboard files for better
        if k_folds > 1:
            self.fold = fold
            self.k_folds = k_folds
            self.writer = SummaryWriter(
                comment=f"_{config_path.stem}_{self.config['eval_params']['batch_size']}_fold: "
                        f"{self.fold}/{self.k_folds}")
        else:
            self.writer = SummaryWriter(
                comment=f"_{config_path.stem}_{self.config['eval_params']['batch_size']}")
        self._model.to(DEVICE)
        if state_dict is not None:
            self._model.load_state_dict(state_dict)
        self.criterion = nn.CrossEntropyLoss()


        self.train_loss_list = []
        self.val_loss_list = []
        self.val_loss = 1000 # initial threshold for dict saving
        self.best_model_dict = {}

        self.val_accuracy = []

        self.total_batch_id = 1
        self.epoch_trained = 0

        self.train_set: bool
    @classmethod
    def load_from_dict(cls, config_path: Path, state_dict: dict):
        return cls(config_path, state_dict, tensorboard=True)
    def state_dict(self) -> dict:
        return self.best_model_dict

    def _load_config(self) -> None:
        with self.config_path.open(mode="r") as yaml_file:
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

    def init_model(self):
        # TODO: make in_channels as parameter
        match self.model_config["type"]:
            case "MLP":
                return networks.MLP(self.layers_configs)
            case "Inception time" | "Inception" | "Inception_time":
                return networks.InceptionTime(self.layers_configs)
            case "LSTM" | "GRU":
                # return networks.RNN(self.layers_configs)
                return networks.RNN(self.layers_configs, attention=self.model_config["attention"])
            case "CNN":
                return networks.CNN(self.layers_configs)
            case "LSTM-FCN" | "lstm_fcn":
                return networks.RnnFcn(self.layers_configs)

    def predict(self, x: np.ndarray, argmax=True) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).to(DEVICE)

            if x.ndim == 2:
                x = torch.reshape(x, (1, 1, -1))
            output = self._model(x)
            if argmax==True:
                return torch.argmax(output, dim=1).cpu().numpy()
            else:
                return output.cpu().numpy()

    # train and validation dataloader split for cases with kfold CV and without CV
    def _train_val_dl_split(self, dataset: Dataset, train_idx: list, val_idx: list) -> (DataLoader, DataLoader):
        if train_idx is not None:
            traindl = DataLoader(dataset,
                                 **self.config["training_params"].get("dataloader_params", {}),
                                 sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            valdl = DataLoader(dataset,
                               **self.config["eval_params"],
                               sampler=torch.utils.data.SubsetRandomSampler(val_idx))
        else:
            traindl = DataLoader(dataset,
                                 **self.config["training_params"].get("dataloader_params", {}),
                                 shuffle=True)
            valdl = DataLoader(dataset,
                               **self.config["eval_params"])
        return traindl, valdl

    def train(self, train_dataset: Dataset, train_idx: list = None, val_idx: list = None) -> None:

        self._load_config()

        optimizer = optim.Adam(self._model.parameters(), self.config["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.config["training_params"][
                                                                   "epoch_num"])

        train_dataloader, val_dataloader = self._train_val_dl_split(train_dataset, train_idx, val_idx)

        epochs = trange(self.config["training_params"]["epoch_num"], ncols=100)  # , desc='Epoch #', leave=True)
        running_loss = 0

        for epoch in epochs:
            # for epoch in range(self.config["training_params"]["epoch_num"]):
            running_val_loss = 0
            for i, (inputs, targets) in enumerate(train_dataloader):
                self._model.train()
                inputs = torch.from_numpy(self._transform(inputs.numpy()))
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()

                outputs = self._model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # calc of the loss, 4 times per epoch
                running_loss += loss.item()
                if (i + 1) % 50 == 0:
                    avg_loss = running_loss / np.ceil(
                        self.config["training_params"]["dataloader_params"]["batch_size"] // 4)
                    self.train_loss_list.append(avg_loss)
                    self.writer.add_scalar(tag=f'Loss/train', scalar_value=avg_loss, global_step=self.total_batch_id)
                    running_loss = 0

                if self.tensorboard:
                    self.train_set = False
                    self.calculate_metrics(val_dataloader)
                    # self.train_set = True
                    # self.calculate_metrics(train_dataset)

                epochs.set_description(f"Epoch #{self.epoch_trained + 1}")
                self.total_batch_id += 1

            # this part of the code serves as check if tensorboard is working correctly
            if self.validation_loss_check:
                j = 0
                for i, (val_inputs, val_targets) in enumerate(val_dataloader):
                    self._model.eval()
                    with torch.no_grad():
                        val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)
                        val_outputs = self.infer(val_inputs)
                        val_loss = self.criterion(val_outputs, val_targets)
                        running_val_loss += val_loss
                        j += 1
                    avg_val_loss = running_val_loss / (j + 1)
                print(f' LOSS train: {avg_loss},  valid: {avg_val_loss}')

            last_lr = scheduler.get_last_lr()[0]  # get the last learning rate
            self.writer.add_scalar(tag='learning rate', scalar_value=last_lr, global_step=self.epoch_trained)
            self.epoch_trained += 1
            scheduler.step()  #

            self.writer.close()
            # epochs.refresh()
            #
            # self.eval_model(testing_data, writer, )




    def calculate_metrics(self, dataloader: DataLoader) -> None:

        if (self.total_batch_id % self.config["tensorboard_params"]["confusion_matrix"] == 0 or
                self.total_batch_id % self.config["tensorboard_params"]["accuracy"] == 0 or
                self.total_batch_id % self.config["tensorboard_params"]["validation_loss"] == 0):
            self._model.eval()

            # concat of tensors is faster than extending lists
            tag = "train" if self.train_set else "val"
            with torch.no_grad():
                outputs = torch.empty(size=(0, 6), dtype=torch.float32, device=DEVICE)
                targets = torch.empty(size=(0, 1), dtype=torch.long, device=DEVICE).flatten()
                for i, (input, target) in enumerate(dataloader):
                    input = torch.from_numpy(self._transform(input.numpy()))
                    input, target = input.to(DEVICE), target.to(DEVICE)
                    output = self._model(input)
                    outputs = torch.cat((outputs, output), dim=0)
                    targets = torch.cat((targets, target), dim=0)
                if not self.train_set and self.total_batch_id % self.config["tensorboard_params"][
                    "validation_loss"] == 0:
                    new_val_loss = self.criterion(outputs, targets)
                    if new_val_loss < self.val_loss:
                        self.best_model_dict = self._model.state_dict()
                    self.val_loss = new_val_loss
                    self.val_loss_list.append(new_val_loss)
                self.writer.add_scalar(tag=f'Loss/val', scalar_value=new_val_loss, global_step=self.total_batch_id)

            class_num = 6

            outputs, targets = np.asarray(outputs.cpu()), np.asarray(targets.cpu())
            predictions = np.argmax(outputs, axis=1)  # makes the correct predictions

            if self.total_batch_id % self.config["tensorboard_params"]["accuracy"] == 0:
                # cr = classification_report(targets, predictions, labels=np.arange(class_num), output_dict=True,
                #                            zero_division=0)
                # self.writer.add_scalar(tag=f'Precision/{tag}',
                #                        scalar_value=cr["macro avg"]["precision"], global_step=self.total_batch_id)
                # self.writer.add_scalar(tag=f'Recall/{tag}',
                #                        scalar_value=cr["macro avg"]['recall'], global_step=self.total_batch_id)
                # self.writer.add_scalar(tag=f'F1-score/{tag}',
                #                        scalar_value=cr["macro avg"]["f1-score"], global_step=self.total_batch_id)
                accuracy = accuracy_score(predictions, targets)
                if not self.train_set:
                    self.val_accuracy.append(accuracy)
                self.writer.add_scalar(tag=f'Accuracy/{tag}',
                                       scalar_value=accuracy, global_step=self.total_batch_id)

            if self.total_batch_id % self.config["tensorboard_params"]["confusion_matrix"] == 0:
                classes = ["LL", "CR", "OT", "MIR", "OL", "N"]
                plt.figure(figsize=(10, 6))
                cm = confusion_matrix(targets, predictions, labels=np.arange(class_num))
                df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in range(class_num)],
                                     columns=[i for i in range(class_num)])
                sns_cm = sns.heatmap(df_cm, annot= True, xticklabels=classes, yticklabels=classes, fmt=".1f")
                self.writer.add_figure(tag=f"Confusion matrix/{tag}",
                                       figure=sns_cm.get_figure(),
                                       global_step=self.total_batch_id)

    def save(self, path: str) -> None:
        torch.save(self._model, Path(path))

