import os
import pickle
import re
import tempfile

import numpy as np

import torch
import os.path

from matplotlib import pyplot as plt
from ray.air import CheckpointConfig
from ray.tune.search.hebo import HEBOSearch
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import seaborn as sns
import networks
from signal_dataset import SignalDataset
import tsaug
import torchaudio.transforms as T
from ray.tune.search.optuna import OptunaSearch
import torchaudio
import ray
from ray import train, tune
from ray.train import Checkpoint, get_checkpoint
import ray.train.torch
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from IPython.display import Audio
from matplotlib.patches import Rectangle
from signal_model import SignalModel, NeuroNet, load_yaml
from pathlib import Path
import torch.optim
from sklearn.model_selection import KFold
import yaml

def update_layer_argument(nn_config: dict, layer_id: str, arg: str, value):
    layers_configs = nn_config["model"]["kwargs"]["layers"]
    if isinstance(layers_configs, list):
        for layer in layers_configs:
            if layer.get("id") == layer_id:
                layer["kwargs"][arg] = value

    elif isinstance(layers_configs, dict):
        for name, layer_config in layers_configs.items():
            for layer in layer_config:
                if layer.get("id") == layer_id:
                    layer["kwargs"][arg] = value

# test_set = SignalDataset(step=10000, window_size=1000, bin_setup=test_config, source_dtype="float32")
def train_network(config, train_config, test_config):


    train_set = SignalDataset(step=5000, window_size=1000, bin_setup=train_config, source_dtype="float32")

    nn_config["training_params"]["lr"] = config["lr"]
    nn_config["training_params"]["dataloader_params"]["batch_size"] = config["batch_size"]
    # nn_config["training_params"]["epoch_num"] = config["epoch_num"]

    update_layer_argument(nn_config, "inceptionblock1", arg="n_filters",
                          value=config["inceptionblock1"]["n_filters"])

    update_layer_argument(nn_config, "inceptionblock1", arg="bottleneck_channels",
                          value=config["inceptionblock1"]["bottleneck_channels"])


    update_layer_argument(nn_config, "inceptionblock2", arg="in_channels",
                          value=4*(config["inceptionblock1"]["n_filters"]))

    update_layer_argument(nn_config, "inceptionblock2", arg="n_filters",
                          value=config["inceptionblock2"]["n_filters"])

    update_layer_argument(nn_config, "inceptionblock2", arg="bottleneck_channels",
                          value=config["inceptionblock2"]["bottleneck_channels"])


    update_layer_argument(nn_config, "adaptivepool", arg="output_size", value=config["adaptivepool"])
    update_layer_argument(nn_config, "linear1", arg="in_features",
                          value=(4*config["inceptionblock2"]["n_filters"]*config["adaptivepool"]))
    update_layer_argument(nn_config, "linear1", arg="out_features", value=config["linear1"])
    update_layer_argument(nn_config, "linear2", arg="in_features", value=config["linear1"])


    neuro_net = NeuroNet(config=nn_config, tensorboard=True)

    neuro_net.optimizer = optim.Adam(neuro_net._model.parameters(), lr=neuro_net.config["training_params"]["lr"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer, nn_config["training_params"]["epoch_num"]))

    train_dl, val_dl = neuro_net._train_val_dl_split(train_set, train_idx=None, val_idx=None)

    # checkpoint = get_checkpoint()
    # if checkpoint:
    #     with checkpoint.as_directory() as checkpoint_dir:
    #         data_path = Path(checkpoint_dir) / "data.pkl"
    #         with open(data_path, "rb") as fp:
    #             checkpoint_state = pickle.load(fp)
    #         start_epoch = checkpoint_state["epoch"]
    #         neuro_net._model.load_state_dict(checkpoint_state["net_state_dict"])
    #         neuro_net.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    # else:
    #     start_epoch = 0
    start_epoch = 0
    running_loss = 0.0


    for epoch in range(start_epoch, nn_config["training_params"]["epoch_num"]):

        neuro_net.train_one_epoch(train_dl, running_loss)
        neuro_net.validate(val_dl)
        scheduler.step()

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": neuro_net._model.state_dict(),
            "optimizer_state_dict": neuro_net.optimizer.state_dict(),
        }
        train.report({"loss": neuro_net.val_loss, "accuracy": neuro_net.val_accuracy[-1],
                     "lr": scheduler.get_last_lr()[0]})
    # with tempfile.TemporaryDirectory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "wb") as fp:
    #         pickle.dump(checkpoint_data, fp)
    #     checkpoint = Checkpoint.from_directory(checkpoint_dir)
    #     train.report({"loss": neuro_net.val_loss, "accuracy": neuro_net.val_accuracy[-1]}, checkpoint=checkpoint)



loaded_signal = []
sample_rate = 1562500
channel = 'ch2'
signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"
train_config = [{"label": (int(i.stem)-1)//4,
              "channels": len(list(i.glob('*' + channel + '.bin'))),
              "interval": [0, int(4.5*sample_rate)],
              "bin_path": list(i.glob('*' + channel + '.bin'))[0]}
             for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
test_config = [{"label": (int(i.stem)-1)//4,
              "channels": len(list(i.glob('*' + channel + '.bin'))),
              "interval": [int(4.5*sample_rate), 5*sample_rate],
              "bin_path": list(i.glob('*' + channel + '.bin'))[0]}
             for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]



config = {
    "inceptionblock1": {
        "n_filters": tune.randint(2, 20),
        "bottleneck_channels": tune.randint(1, 16),
        # "kernel_sizes": [tune.choice([2*i+1 for i in range(7, 15)]),
        #                  tune.choice([2*i+1 for i in range(25, 45)]),
        #                  tune.choice([2*i+1 for i in range(45, 80)])]
    },
    "inceptionblock2": {
        "n_filters": tune.randint(16, 50),
        "bottleneck_channels": tune.randint(16, 50),
        # "kernel_sizes": [tune.choice([2 * i + 1 for i in range(7, 15)]),
        #                  tune.choice([2 * i + 1 for i in range(25, 45)]),
        #                  tune.choice([2 * i + 1 for i in range(45, 80)])]
    },
    "adaptivepool": tune.randint(6, 15),
    "linear1": tune.randint(256, 1024),
    "lr": tune.uniform(1e-5, 1e-1),
    "batch_size": tune.choice([128, 256, 512, 755, 1024]),
    # "epoch_num": tune.choice([10, 15, 20])
}

network = "InceptionTime"
nn_config = load_yaml(Path("nn_yaml_configs/" + network + ".yaml"))
hebo = HEBOSearch(metric="loss", mode="min")
hebo.restore("/home/petr/ray_results/train_network_2024-06-13_22-10-15/searcher-state-2024-06-13_22-10-15.pkl")

optuna_search = OptunaSearch(
    metric="accuracy",
    mode="max")

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='loss',
    mode='min',
    max_t=20,
    grace_period=5,
    reduction_factor=3
)
 # ②
result = tune.run(
    partial(train_network, train_config=train_config, test_config=test_config),
    resources_per_trial={"cpu": 10, "gpu": 1},
    num_samples=150,
    checkpoint_config=CheckpointConfig(num_to_keep=5),
    search_alg=hebo,
    scheduler=asha_scheduler,
    verbose=1,
)


best_trial = result.get_best_trial("accuracy", "max", "last")
print(best_trial)
print(f"Best trial accuracy: {best_trial.last_result['accuracy']}")
print(f"Best trial validation loss: {best_trial.last_result['loss']}")

print(f"Best trial config: {best_trial.config}")

# │ adaptivepool                                 10 │
# │ batch_size                                  256 │
# │ inceptionblock1/bottleneck_channels           4 │
# │ inceptionblock1/n_filters                    15 │
# │ inceptionblock2/bottleneck_channels          47 │
# │ inceptionblock2/n_filters                    20 │
# │ linear1                                     305 │
# │ lr                                      0.00251