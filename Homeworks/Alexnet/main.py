from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn  # all neural network moduls, nn.Linear, nn.Conv2D. BatchNorm, Loss fctions
import torch.nn.functional as F  # All functions that dont have parameters
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms  # Transformation we can perform on our dataset
import yaml
# import os
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data import Dataset  # Gives easier dataset managment and creates mini batches
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from tqdm import tqdm

import imgaug as ia
import imgaug.augmenters as iaa

DEVICE = 'cpu'
# print(f'Using {DEVICE} for inference')
torch.manual_seed(2)


class CNNOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, conv_layers: list[dict], fullycon_layers: list[dict]):
        super().__init__()
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.fullycon_layers = nn.ModuleList(fullycon_layers)

    def forward(self, x):

        for layer in self.conv_layers:
            # residual = x
            x = F.relu(layer(x))
            # x += residual

        x = torch.flatten(x, 1)

        for layer in self.fullycon_layers[:-1]:
            x = F.relu(layer(x))

        return self.fullycon_layers[-1](x)


class NeuroNet:
    def __init__(self, control_center: Path, meta: dict = ''):
        self._load_yaml(control_center)
        # self.model = self.build_old_model()
        self.model = self.build_model()
        self.model.to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.training_params["epoch_num"])
        self.criterion = nn.CrossEntropyLoss()

    def _load_yaml(self, yaml_path: Path) -> None:

        path = Path(yaml_path)
        with path.open(mode="r") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.SafeLoader)

        self.classes = data["classes"]
        self.lr = data["lr"]
        self.training_params = data["training_params"]
        self.eval_params = data["eval_params"]

        # load convolutional layers
        self.conv_layers_config = data['conv_layers']
        self.conv_layers = []
        for layer_config in self.conv_layers_config:
            self.conv_layers.append(
                eval('nn.' + layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))
        # load fully connected layers
        self.fullycon_layers_config = data['fully-con_layers']
        self.fullycon_layers = []
        for layer_config in self.fullycon_layers_config:
            self.fullycon_layers.append(
                eval('nn.' + layer_config['name'])(*layer_config.get('args', []), **layer_config.get('kwargs', {})))

    def build_model(self):
        return CNN(self.conv_layers, self.fullycon_layers)

    def build_old_model(self):
        return CNNOld()

    def train_model(self, training_data: Dataset, test_data: Dataset):
        writer = SummaryWriter(f'runs/CIFAR10/other')
        train_dataloader = DataLoader(training_data, **self.training_params.get('dataloader_params', {}))
        step = 0
        for epoch in range(self.training_params["epoch_num"]):
            self.model.train()

            for batch_idx, (inputs, labels) in enumerate(
                    tqdm(train_dataloader, desc=f'epoch {epoch + 1} / {self.training_params["epoch_num"]}')):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                writer.add_scalar('Training Loss', loss, global_step=step)
                step += 1
            # EVAL_____________________________________________________________________________________________________

            self.model.eval()
            test_dataloader = DataLoader(test_data, **self.eval_params)
            cm = np.zeros((10, 10))

            with torch.no_grad():
                for (inputs, labels) in test_dataloader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    # predictions= predictions.cpu()
                    cm_p = confusion_matrix(labels, predictions, labels=np.arange(10))
                    cm = cm + cm_p

            correct_pred = 0
            total_pred = 0

            for i in range((len(cm[0]))):
                for j in range((len(cm[0]))):
                    if i == j:
                        correct_pred = correct_pred + cm[i][j]
                        total_pred = total_pred + cm[i][j]
                    else:
                        total_pred = total_pred + cm[i][j]

            accuracy = correct_pred / total_pred
            writer.add_scalar('Training Accuracy', accuracy, global_step=epoch)
            last_lr = self.scheduler.get_last_lr()[0]  # WHY IS IT A LIST
            writer.add_scalar('learning rate', last_lr, global_step=epoch)

            # writer.add_hparams({'learning rate': last_lr, 'bsize': batch_size},
            #                    {'accuracy': accuracy, 'loss': loss})
            print(f"\n accuracy: {accuracy}")
            self.scheduler.step()
        print(cm)
        writer.close()

    # ________________________________________________________________________________________________________________________

    def eval_model(self, test_data: DataLoader):

        test_dataloader = DataLoader(test_data, **self.eval_params)
        self.model.eval()
        cm = np.zeros((10, 10))

        with torch.no_grad():
            for (inputs, true_labels) in tqdm(test_dataloader):
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)
                predictions = np.array(predictions)
                cm_p = confusion_matrix(true_labels, predictions, labels=np.arange(10))
                cm = cm + cm_p

        cm_collumn_sum = cm.sum(axis=0)
        # print(cm_collumn_sum)
        # cm_percent= cm/cm_collumn_sum
        # cm_percent= np.round(np.where(np.isnan(cm_percent), 0, cm_percent), decimals= 3)
        # print(cm_percent)

        plt.matshow(cm)
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

    def precision_per_class(self, test_data: DataLoader):

        test_dataloader = DataLoader(test_data, **self.eval_params)

        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        self.model.eval()  # important to switch behavior of model (batchnorm, dropout.. disabled)

        with torch.no_grad():
            for (inputs, labels) in tqdm(test_dataloader):
                outputs = self.model(inputs)

                # print(torch.max(outputs, 1))
                # assert False   stops the code

                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():  # .items() makes iteration through keys and vals
            accuracy = 100 * correct_count / total_pred[classname]
            print(f'Precision for class: {classname:5s} is {accuracy:.1f} %')

    def predict(self, img: torch.Tensor):
        self.model.eval()
        xb = img.unsqueeze(0)
        yb = self.model(xb)
        _, predictions = torch.max(yb, dim=1)
        return self.classes[predictions[0]]

    def save_model(self):
        PATH = './cifar_mnist.pth'
        torch.save(self.model.state_dict(), PATH)


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose([
    np.asarray,
    iaa.Sequential([
        iaa.flip.Fliplr(p=0.5),
        # iaa.flip.Flipud(p=0.5),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                   rotate=(-15, 15),
                   scale=(0.5, 1.5)),
        # iaa.MultiplyBrightness(mul=(0.65, 1.35)),
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0.0, 2)),
        )
    ]).augment_image,
    # np.copy,
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

neuro_net = NeuroNet(Path('alexnet_3conv2lin.yaml'))

neuro_net.train_model(training_data, test_data)

neuro_net.eval_model(test_data)

neuro_net.precision_per_class(test_data)

neuro_net.save_model()
inputs, label = test_data[0]
print(label)
print(neuro_net.classes)
print('Label:', neuro_net.classes[label], ', Predicted:', neuro_net.predict(img))
