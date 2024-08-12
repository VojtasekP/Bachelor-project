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


    neuro_net = NeuroNet(config=nn_config, metrics=True)

    match network:
        case "InceptionTime":
            neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(), **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
                                                       T_max=nn_config["training_params"]["epoch_num"])

            # cutoff = {2500: sample_rate // 2, 5000: sample_rate//3, 10000: sample_rate//5, 15000:sample_rate//8, 20000: sample_rate//10, 25000: sample_rate//13, 30000: sample_rate//15}

        case "LSTM":
            neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(), **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
                                                       T_max=nn_config["training_params"]["epoch_num"])
            # cutoff = {2500: sample_rate , 5000: sample_rate//2, 10000: sample_rate//4, 15000:sample_rate//6, 20000: sample_rate//8, 25000: sample_rate//10, 30000: sample_rate//12}
        case "CNN":

            neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(), **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
                                                       T_max=nn_config["training_params"]["epoch_num"])

            # cutoff = {2500: sample_rate, 5000: sample_rate, 10000: sample_rate, 15000:sample_rate, 20000: sample_rate, 25000: sample_rate, 30000: sample_rate}

    augmentation_params = {
         'add_noise': config["add_noise"],  # Add Gaussian noise with 0.0001 standard deviation
         'frequency_shift': config["frequency_shift"],
         'scale': config["scale"],  # Scale by up to 10%
         'time_stretch': config["time_stretch"],  # Stretch/compress by up to 40%
         "magnitude_shift": config["magnitude_shift"]
    }
    train_set = SignalDataset(step=10000, window_size=30000, bin_setup=train_config, aug_params=augmentation_params, cutoff=sample_rate//15, source_dtype="float32")
    val_set = SignalDataset(step=10000, window_size=30000, bin_setup=val_config, cutoff=sample_rate//15, source_dtype="float32")
    train_dl = DataLoader(train_set, **nn_config["training_params"].get("dataloader_params", {}), shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_set, **nn_config["eval_params"], pin_memory=True)

    start_epoch = 0
    running_loss = 0.0
    early_stopper = EarlyStopper(patience=12)

    for epoch in range(start_epoch, nn_config["training_params"]["epoch_num"]):
        neuro_net.train_one_epoch(train_dl, running_loss)
        neuro_net.validate(val_dl)
        scheduler.step()

        train.report({"loss": neuro_net.val_loss, "accuracy": neuro_net.val_accuracy[-1],
                      "lr": scheduler.get_last_lr()[0]})

        if early_stopper.early_stop(neuro_net.val_loss):
            print(f"Training stopped due to early stopping. Last epoch: {epoch}")
            break


aug = {
    'add_noise': tune.uniform(0, 0.1),  # Add Gaussian noise with 0.0001 standard deviation
    'frequency_shift': tune.randint(0, 10000),
    'scale': tune.uniform(0, 1),  # Scale by up to 10%
    'time_stretch': tune.uniform(0, 1),  # Stretch/compress by up to 40%
    "magnitude_shift": tune.uniform(0.7, 1.3),
}



nn_models = [ "CNN", "LSTM"]

for network in nn_models:

    hebo = HEBOSearch(metric="accuracy", mode="max")
    iter = {"InceptionTime": 51, "LSTM": 51, "CNN": 51}
    iter_min = {"InceptionTime": 15,  "LSTM": 15, "CNN": 15}
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='accuracy',
        mode='max',
        max_t=iter[network],
        grace_period=iter_min[network],
    )

    # ②
    result = tune.run(
        partial(train_network, network=network),
        name="AUG_" + network,
        resources_per_trial={"cpu": 10, "gpu": 1},
        num_samples=5,
        search_alg=hebo,
        scheduler=asha_scheduler,
        config=aug,
        storage_path="/mnt/home2/hparams_checkpoints/",
        verbose=1,
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(best_trial)
    print(f"Best trial accuracy: {best_trial.last_result['accuracy']}")
    print(f"Best trial validation loss: {best_trial.last_result['loss']}")

    print(f"Best trial config: {best_trial.config}")

# tensorboard --logdir /tmp/ray/session_2024-06-16_22-52-02_765634_29559/artifacts/2024-06-16_22-52-05/main_params-InceptionTime/driver_artifacts