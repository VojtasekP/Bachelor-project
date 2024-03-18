from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from icecream import ic
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from signal_dataset import SignalDataset
import networks
import re

DEVICE = 'cuda'
torch.manual_seed(22)






class NeuroNet:
    def __init__(self, control_center: Path):
        self._load_yaml(control_center)
        self.model = self.build_model(control_center)
        self.model.to('cuda')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.training_params["epoch_num"])
        self.criterion = nn.MSELoss()
        self.history = []
        self.mse_avg = 0

    def _load_yaml(self, yaml_path: Path) -> None:

        with yaml_path.open(mode="r") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.SafeLoader)

        self.training_params = data["training_params"]
        self.eval_params = data["eval_params"]
        self.lr = data["lr"]
        self.layers_config = data['layers']

    def build_model(self, yaml_path: Path):
        if yaml_path == Path('yaml_configs/MLP.yaml'):
            return networks.NN(self.layers_config)
        if yaml_path == Path('yaml_configs/InceptionTime.yaml'):
            return networks.InceptionTime(self.layers_config)
        if yaml_path == Path('yaml_configs/LSTM.yaml') or yaml_path == Path('yaml_configs/GRU.yaml'):
            return networks.RNN(self.layers_config)

    def train_model(self, training_data: Dataset, testing_data: Dataset):
        writer = SummaryWriter(comment="_Inception_Time")

        step = 0

        train_dataloader = DataLoader(training_data, **self.training_params.get("dataloader_params", {}))
        test_dataloader = DataLoader(testing_data, **self.eval_params)

        for epoch in range(self.training_params["epoch_num"]):
            self.model.train()
            self.history = []
            for batch_id, (inputs, targets) in enumerate(
                    tqdm(train_dataloader, desc=f'epoch {epoch + 1}/ {self.training_params["epoch_num"]} ')):
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

            # ic(self.scheduler.get_last_lr())
            last_lr = self.scheduler.get_last_lr()[0]
            writer.add_scalar('learning rate', last_lr, global_step=epoch)
            self.scheduler.step()
            print(f"Current loss is {self.mse_avg} ")
            ic("___________________________")
        writer.close()
        # torch.optim.lr_scheduler.print_lr()

    def predict(self, data: torch.Tensor):
        with torch.no_grad():
            points, freq = data
            return print(f'predicted freq: {self.model(points)}, real freq: {freq}')

signal_data_dir = "/home/petr/Documents/Motor_project/AE_PETR_motor/"
sr = 1562500


bin_setup = [{"label": i.stem, "interval": [0, 15*sr], "bin_path": list(i.glob('*.bin'))[0]} for i in Path(signal_data_dir).glob('WUP*') if re.search(r'[\d]$', i.stem)]


sd = SignalDataset(step=1000, window_size=1000, bin_setup=bin_setup, device="cpu", source_dtype="float32")

train_data, test_data = random_split(sd, [0.8, 0.2])

neuro_net = NeuroNet(Path('yaml_configs/InceptionTime.yaml'))

neuro_net.train_model(train_data, test_data)

neuro_net.predict((test_data[0]))
