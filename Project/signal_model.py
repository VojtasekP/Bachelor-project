import abc
import random
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
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import networks.networks as networks

DEVICE = "cuda"

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class SignalModel:
    def __init__(self, config: dict, aug_params: dict = None):
        self.config = config
        self.model_config = self.config["model"]
        self.classes = self.config["classes"]
        self.aug_params = aug_params
        self._model = self.init_model()
        self._train_transform = self.init_transform(train=True)
        self._val_transform = self.init_transform(train=False)

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

    def init_transform(self, train:bool) -> Callable:

        def transform(x):
            return x
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

    def __init__(self, config: dict, aug_params: dict=None, state_dict: dict = None, metrics: bool = False):
        super().__init__(config, aug_params)


        self.metrics = metrics
        self.writer = SummaryWriter(
            comment=f"_{self.model_config['class']}_{self.config['training_params']['dataloader_params']['batch_size']}")

        self._model.to(DEVICE)
        if state_dict is not None:
            self._model.load_state_dict(state_dict)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.best_model_dict = {}

        self.train_loss_list = []
        self.val_loss_list = []
        self.val_loss = np.infty
        self.val_accuracy = []

        self.best_loss = 1000
        self.total_batch_id = 1
        self.epoch_trained = 0
        self.train: bool
        self.total_params = sum(p.numel() for p in self._model.parameters())
        self.trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)


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
        self.load_layers()
        match self.config["model"]["class"]:
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
            case "ResNet":
                return networks.ResNet(self.layers_configs, num_classes=6)
            case "EfficientNet":
                return networks.EfficientNet(self.layers_configs, num_classes=6)


    def _init_optimizer(self, optimizer: str):
        optimizer = optimizer.lower()
        match optimizer:
            case "sgd":
                return optim.SGD(self._model.parameters(), **self.config["training_params"]["optimizer_params"].get("kwargs", {}))
            case "adam":
                return optim.Adam(self._model.parameters(), **self.config["training_params"]["optimizer_params"].get("kwargs", {}))
            case "adamw":
                return optim.AdamW(self._model.parameters(), **self.config["training_params"]["optimizer_params"].get("kwargs", {}))
            case "rmsprop":
                return optim.RMSprop(self._model.parameters(), **self.config["training_params"]["optimizer_params"].get("kwargs", {}))

    def _init_scheduler(self, scheduler: str):
        scheduler = scheduler.lower()
        match scheduler:
            case "cosineAnnealing"| "cosine" | "cosine_annealing":
                return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.config["training_params"]["scheduler_params"].get("kwargs", {}))
                # T_max
            case "exponential":
                return optim.lr_scheduler.ExponentialLR(self.optimizer, **self.config["training_params"]["scheduler_params"].get("kwargs", {}))
                # gamma
            case "cosineAnnealingWithRestarts" | "warmups" | "cosine_warmup" | "cosine_annealing_warmup":
                return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **self.config["training_params"]["scheduler_params"].get("kwargs", {}))
                # T_0
            case "polynomial":
                return optim.lr_scheduler.PolynomialLR(self.optimizer, **self.config["training_params"]["scheduler_params"].get("kwargs", {}))
                # power and total_iter



    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict = None, patience:int = 5) -> None:
        if config is not None:
            self.config = config
        self.optimizer = self._init_optimizer(optimizer=self.config["training_params"]["optimizer_params"]["class"])
        scheduler = self._init_scheduler(scheduler=self.config["training_params"]["scheduler_params"]["class"])


        epochs = trange(self.config["training_params"]["epoch_num"], ncols=100)  # , desc='Epoch #', leave=True)
        running_loss = 0
        early_stopper = EarlyStopper(patience=patience)


        for epoch in epochs:
            self.train_one_epoch(train_dataloader, running_loss)
            self.validate(val_dataloader)
            epochs.set_description(f"Epoch #{self.epoch_trained + 1}")

            last_lr = scheduler.get_last_lr()[0]  # get the last learning rate
            self.writer.add_scalar(tag='learning rate', scalar_value=last_lr, global_step=self.epoch_trained)
            scheduler.step()

            if early_stopper.early_stop(self.val_loss):
                print(f"Training stopped due to early stopping. Last epoch: {self.epoch_trained}")
                break
        self.writer.close()
        print(f"Best loss achieved at epoch {self.best_epoch}")

    def train_one_epoch(self, train_dataloader, running_loss):
        outputs = torch.empty(size=(0, 6), dtype=torch.float32, device=DEVICE).flatten()
        targets = torch.empty(size=(0, 1), dtype=torch.long, device=DEVICE).flatten()
        for i, (input, target) in enumerate(train_dataloader):
            self._model.train()
            input = torch.from_numpy(self._train_transform(input.numpy()))
            input, target = input.to(DEVICE), target.to(DEVICE)
            self.optimizer.zero_grad()

            output = self._model(input)
            outputs = torch.cat((outputs, output), dim=0)
            targets = torch.cat((targets, target), dim=0)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                avg_loss = running_loss / 50
                self.train_loss_list.append(avg_loss)
                self.writer.add_scalar(tag=f'Train_loss', scalar_value=avg_loss, global_step=self.total_batch_id)
                running_loss = 0

            self.total_batch_id += 1
        self.epoch_trained += 1
        if self.metrics:
            self._model.eval()
            with torch.no_grad():
                self.calculate_metrics(outputs, targets, tag="train")
    # VALIDATION
    def validate(self, val_dataloader: DataLoader):
        self._model.eval()
        num_batches = len(val_dataloader)
        with torch.no_grad():
            # concatenation of tensors is faster than extending lists
            outputs = torch.empty(size=(0, 6), dtype=torch.float32, device=DEVICE).flatten()
            targets = torch.empty(size=(0, 1), dtype=torch.long, device=DEVICE).flatten()
            running_loss = 0.0
            for i, (input, target) in enumerate(val_dataloader):
                input, target = input.to(DEVICE), target.to(DEVICE)
                output = self._model(input)
                running_loss += self.criterion(output, target).item()
                outputs = torch.cat((outputs, output), dim=0)
                targets = torch.cat((targets, target), dim=0)

            if self.metrics:
                self.calculate_metrics(outputs, targets, tag="val")
            new_val_loss = running_loss/ num_batches
            if new_val_loss < self.best_loss:
                self.best_model_dict = self._model.state_dict()
                self.best_epoch = self.epoch_trained + 1
                self.best_loss = new_val_loss

            self.val_loss_list.append(new_val_loss)
            self.writer.add_scalars(main_tag=f'Loss', tag_scalar_dict={"val": new_val_loss, "train": self.train_loss_list[-1]}, global_step=self.total_batch_id)
            self.val_loss = new_val_loss

    def predict(self, x: np.ndarray, argmax=True) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            x = self._val_transform(x)
            x = torch.from_numpy(x).to(DEVICE)

            if x.ndim == 2:
                x = torch.reshape(x, (1, 1, -1))
            output = self._model(x)

            if argmax:
                return torch.argmax(output, dim=1).cpu().numpy()
            else:
                return torch.softmax(output, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def calculate_metrics(self, outputs, targets, tag:str) -> None:
        outputs, targets = np.asarray(outputs.cpu()), np.asarray(targets.cpu())
        predictions = np.argmax(outputs, axis=1)

        # CONFUSION MATRIX PLOT
        plt.figure(figsize=(10, 6))
        cm = confusion_matrix(targets, predictions, labels=np.arange(len(self.classes)))
        df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in range(len(self.classes))],
                             columns=[i for i in range(len(self.classes))])
        sns_cm = sns.heatmap(df_cm, annot=True, xticklabels=self.classes, yticklabels=self.classes, fmt=".1f")

        self.writer.add_figure(tag=f"Confusion matrix/{tag}",
                               figure=sns_cm.get_figure(),
                               global_step=self.total_batch_id)

        # F1-SCORE
        cr = classification_report(targets, predictions,
                                   labels=np.arange(len(self.classes)),
                                   output_dict=True,
                                   zero_division=0)

        self.writer.add_scalar(tag=f'F1-score/{tag}',
                               scalar_value=cr["weighted avg"]["f1-score"],
                               global_step=self.total_batch_id)

        self.writer.add_scalar(tag=f'Precision/{tag}',
                               scalar_value=cr["weighted avg"]["precision"],
                               global_step=self.total_batch_id)

        self.writer.add_scalar(tag=f'Recall/{tag}',
                               scalar_value=cr["weighted avg"]["recall"],
                               global_step=self.total_batch_id)


        # ACCURACY
        accuracy = accuracy_score(predictions, targets)

        self.writer.add_scalar(tag=f'Accuracy/{tag}',
                               scalar_value=accuracy,
                               global_step=self.total_batch_id)
        if tag == "val":
            self.val_accuracy.append(accuracy)