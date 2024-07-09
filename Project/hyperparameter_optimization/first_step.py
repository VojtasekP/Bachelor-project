import re

import torch

from ray.tune.search.hebo import HEBOSearch
from torch import optim
from torch.utils.data import DataLoader

from dataset.signal_dataset import SignalDataset
from ray import train, tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from signal_model import NeuroNet, EarlyStopper
from pathlib import Path
import torch.optim
import yaml


def load_yaml(config_path: Path):
    with config_path.open(mode="r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.SafeLoader)


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
def train_network(config, network):
    sample_rate = 1562500
    signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"
    train_config = [{"label": (int(i.stem) - 1) // 4,
                     "channels": len(list(i.glob('*ch2.bin'))),
                     "interval": [0, int(4 * sample_rate)],
                     "bin_path": list(i.glob('*ch2.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    val_config = [{"label": (int(i.stem) - 1) // 4,
                     "channels": len(list(i.glob('*ch2.bin'))),
                     "interval": [int(4 * sample_rate), int(4.5 * sample_rate)],
                     "bin_path": list(i.glob('*ch2.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    nn_config = load_yaml(Path("/home/petr/Documents/bachelor_project/Project/configs/nn_configs/" + network + ".yaml"))

    train_set = SignalDataset(step=10000, window_size=5000, bin_setup=train_config, source_dtype="float32")
    val_set = SignalDataset(step=10000, window_size=5000, bin_setup=val_config, source_dtype="float32")
    neuro_net = NeuroNet(config=nn_config, metrics=True)

    match config["optimizer"]:
        case "adam":
            neuro_net.optimizer = optim.Adam(neuro_net._model.parameters(), lr=config["lr"],
                                             weight_decay=config["weight_decay"])
        case "adamw":
            neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(), lr=config["lr"],
                                              weight_decay=config["weight_decay"])
        case "rmsprop":
            neuro_net.optimizer = optim.RMSprop(neuro_net._model.parameters(), lr=config["lr"],
                                                weight_decay=config["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
                                                     T_max=nn_config["training_params"]["epoch_num"])

    train_dl = DataLoader(train_set, **nn_config["training_params"].get("dataloader_params", {}), shuffle=True,
                         pin_memory=True)

    val_dl = DataLoader(val_set, **nn_config["eval_params"], pin_memory=True)
    start_epoch = 0
    running_loss = 0.0
    early_stopper = EarlyStopper(patience=15)
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
                      "lr": scheduler.get_last_lr()[0], "epoch_trained": epoch})

        if early_stopper.early_stop(neuro_net.val_loss):
            print(f"Training stopped due to early stopping. Last epoch: {epoch}")
            break


config = {
    "lr": tune.uniform(1e-6, 1e-2),
    "optimizer": tune.choice(["adam", "rmsprop", "adamw"]),
    "weight_decay": tune.uniform(1e-4, 1e-1),
}

nn_models = ["InceptionTime"]

for network in nn_models:
    hebo = HEBOSearch(metric="accuracy", mode="max")
    iter = {"InceptionTime": 51, "LSTM": 51, "CNN": 51}
    iter_min = {"InceptionTime": 20, "LSTM": 20, "CNN_spec": 20, "CNN": 20}
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='accuracy',
        mode='max',
        max_t=iter[network],
        grace_period=iter_min[network]
    )

    # ②
    result = tune.run(
        partial(train_network, network=network),
        name="FIRST_STEP-" + network,
        resources_per_trial={"cpu": 10, "gpu": 1},
        num_samples=100,
        search_alg=hebo,
        scheduler=asha_scheduler,
        config=config,
        storage_path="/mnt/home2/hparams_checkpoints/",
        verbose=1,
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(best_trial)
    print(f"Best trial accuracy: {best_trial.last_result['accuracy']}")
    print(f"Best trial validation loss: {best_trial.last_result['loss']}")

    print(f"Best trial config: {best_trial.config}")

# tensorboard --logdir /tmp/ray/session_2024-06-16_22-52-02_765634_29559/artifacts/2024-06-16_22-52-05/main_params-InceptionTime/driver_artifacts
