import abc
from abc import ABC
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import tsaug
import yaml
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import networks

DEVICE = "cuda"


def load_yaml(config_path: Path) -> dict:
    with config_path.open(mode="r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.SafeLoader)


class SignalModel:
    def __init__(self, config: dict):
        self.config = config
        self.model_config = self.config["model"]
        self._model = self.init_model()
        self._transform = self.init_transform()

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
        class_num = 6  # in future will not be hard coded
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(class_num))
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, cmap='Greens')
        for i in range(class_num):
            for j in range(class_num):
                plt.text(j, i, cm[i, j], ha="center", va="bottom", color='gray')
                plt.text(j, i, str(j), ha="center", va="top", color='gray')
        plt.show()

    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    def precision_and_recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> (float, float):
        cr = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        return cr["macro avg"]["precision"], cr["macro avg"]["recall"]

    def classification_report(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        return classification_report(y_true, y_pred, zero_division=0, output_dict=True)


class SklearnModel(SignalModel, ABC):

    def __init__(self, config: dict):
        super().__init__(config)

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

    def __init__(self, config: dict, state_dict: dict = None,
                 tensorboard: bool = False,
                 fold: int = 1, k_folds: int = 1):
        super().__init__(config)
        self.tensorboard = tensorboard
        # in case of CV, changing tensorboard files for better
        if k_folds > 1:
            self.fold = fold
            self.k_folds = k_folds
            self.writer = SummaryWriter(
                comment=f"_{self.model_config['type']}_{self.config['eval_params']['batch_size']}_fold: "
                        f"{self.fold}/{self.k_folds}")
        else:
            self.writer = SummaryWriter(
                comment=f"_{self.model_config['type']}_{self.config['eval_params']['batch_size']}")

        self._model.to(DEVICE)
        if state_dict is not None:
            self._model.load_state_dict(state_dict)

        self.criterion = nn.CrossEntropyLoss()
        self.best_model_dict = {}

        self.train_loss_list = []
        self.val_loss_list = []
        self.val_loss = np.infty
        self.val_accuracy = []

        self.total_batch_id = 1
        self.epoch_trained = 0

    def state_dict(self) -> dict:
        return self.best_model_dict

    def load_layers(self):
        layers = self.model_config["kwargs"]["layers"]
        if isinstance(layers, list):
            self.layers_configs = []
            for layer_config in self.model_config["kwargs"]["layers"]:
                self.layers_configs.append(layer_config)
        else:
            self.layers_configs = {}
            for name, kwargs in self.model_config["kwargs"]["layers"].items():
                self.layers_configs[name] = kwargs

    def init_model(self):
        # TODO: make in_channels as parameter
        self.load_layers()
        match self.config["model"]["type"]:
            case "MLP":
                return networks.MLP(self.layers_configs)
            case "Inception time" | "Inception" | "Inception_time":
                return networks.InceptionTime(self.layers_configs)
            case "LSTM" | "GRU":
                # return networks.RNN(self.layers_configs)
                return networks.RNN(self.layers_configs)
            case "CNN" | "CNN_spec":
                return networks.CNN(self.layers_configs)
            case "LSTM-FCN" | "lstm_fcn":
                return networks.RnnFcn(self.layers_configs)

    # train and validation dataloader split for cases with kfold CV and without CV
    def _train_val_dl_split(self, dataset: Dataset, train_idx, val_idx) -> (DataLoader, DataLoader):
        if train_idx is not None:
            traindl = DataLoader(dataset,
                                 **self.config["training_params"].get("dataloader_params", {}),
                                 sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            valdl = DataLoader(dataset,
                               **self.config["eval_params"],
                               sampler=torch.utils.data.SubsetRandomSampler(val_idx))
        else:
            train_ds, test_ds = random_split(dataset, [0.8889, 0.1111])     # 0.8889 out of 0.9 is ~ 0.8
                                                                                    # so (train, val, test)
                                                                                    #    (  0.8, 0.1, 0.1 ) splits
            traindl = DataLoader(train_ds,
                                 **self.config["training_params"].get("dataloader_params", {}),
                                 shuffle=True)
            valdl = DataLoader(test_ds,
                               **self.config["eval_params"])
        return traindl, valdl

    def train(self, train_dataset: Dataset, train_idx: list = None, val_idx: list = None) -> None:

        self.optimizer = optim.Adam(self._model.parameters(), lr=self.config["training_params"]["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                        T_0=(1+(self.config["training_params"]["epoch_num"])//self.config["training_params"]["warmups"]))

        train_dataloader, val_dataloader = self._train_val_dl_split(train_dataset, train_idx, val_idx)

        epochs = trange(self.config["training_params"]["epoch_num"], ncols=100)  # , desc='Epoch #', leave=True)
        running_loss = 0



        for epoch in epochs:
            running_val_loss = 0
            self.train_one_epoch(train_dataloader, running_loss)
            self.validate(val_dataloader)
            epochs.set_description(f"Epoch #{self.epoch_trained + 1}")

            last_lr = scheduler.get_last_lr()[0]  # get the last learning rate
            self.writer.add_scalar(tag='learning rate', scalar_value=last_lr, global_step=self.epoch_trained)
            scheduler.step()
            self.epoch_trained += 1
        self.writer.close()

    def train_one_epoch(self, train_dataloader, running_loss):

        for i, (inputs, targets) in enumerate(train_dataloader):
            self._model.train()
            inputs = torch.from_numpy(self._transform(inputs.numpy()))
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()

            outputs = self._model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                avg_loss = running_loss / 50
                self.train_loss_list.append(avg_loss)
                self.writer.add_scalar(tag=f'Loss/train', scalar_value=avg_loss, global_step=self.total_batch_id)
                running_loss = 0

            self.total_batch_id += 1

    # VALIDATION
    def validate(self, val_dataloader: DataLoader):
        self._model.eval()
        with torch.no_grad():
            # concatenation of tensors is faster than extending lists
            outputs = torch.empty(size=(0, 6), dtype=torch.float32, device=DEVICE)
            targets = torch.empty(size=(0, 1), dtype=torch.long, device=DEVICE).flatten()
            for i, (input, target) in enumerate(val_dataloader):
                input = torch.from_numpy(self._transform(input.numpy()))
                input, target = input.to(DEVICE), target.to(DEVICE)
                output = self._model(input)
                outputs = torch.cat((outputs, output), dim=0)
                targets = torch.cat((targets, target), dim=0)

            if self.tensorboard:
                self.calculate_metrics(outputs, targets)

            new_val_loss = float(self.criterion(outputs, targets).cpu().numpy())
            if new_val_loss < self.val_loss:
                self.best_model_dict = self._model.state_dict()

            self.val_loss_list.append(new_val_loss)
            self.writer.add_scalar(tag=f'Loss/val', scalar_value=new_val_loss, global_step=self.total_batch_id)
            self.val_loss = new_val_loss

    def calculate_metrics(self, outputs, targets) -> None:

        classes = ["IL", "CD", "EL", "CL", "OS", "N"] #  TODO: FROM YAML
        outputs, targets = np.asarray(outputs.cpu()), np.asarray(targets.cpu())
        predictions = np.argmax(outputs, axis=1)

        # CONFUSION MATRIX PLOT
        plt.figure(figsize=(10, 6))
        cm = confusion_matrix(targets, predictions, labels=np.arange(len(classes)))
        df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in range(len(classes))],
                             columns=[i for i in range(len(classes))])
        sns_cm = sns.heatmap(df_cm, annot=True, xticklabels=classes, yticklabels=classes, fmt=".1f")

        self.writer.add_figure(tag=f"Confusion matrix/val",
                               figure=sns_cm.get_figure(),
                               global_step=self.total_batch_id)

        # F1-SCORE
        cr = classification_report(targets, predictions,
                                   labels=np.arange(len(classes)),
                                   output_dict=True,
                                   zero_division=0)

        self.writer.add_scalar(tag=f'F1-score',
                               scalar_value=cr["macro avg"]["f1-score"],
                               global_step=self.total_batch_id)

        # ACCURACY
        accuracy = accuracy_score(predictions, targets)
        self.val_accuracy.append(accuracy)
        self.writer.add_scalar(tag=f'Accuracy/val',
                               scalar_value=accuracy,
                               global_step=self.total_batch_id)

    def predict(self, x: np.ndarray, argmax=True) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).to(DEVICE)

            if x.ndim == 2:
                x = torch.reshape(x, (1, 1, -1))
            output = self._model(x)

            if argmax:
                return torch.argmax(output, dim=1).cpu().numpy()
            else:
                return torch.softmax(output, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self._model, path)
