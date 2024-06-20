import re

import torch

from ray.tune.search.hebo import HEBOSearch
from torch import optim
from dataset.signal_dataset import SignalDataset
from ray import train, tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from signal_model import NeuroNet, load_yaml
from pathlib import Path
import torch.optim


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
                     "interval": [0, int(4.5 * sample_rate)],
                     "bin_path": list(i.glob('*ch2.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
    nn_config = load_yaml(Path("/home/petr/Documents/bachelor_project/Project/nn_configs/" + network + ".yaml"))

    if network == "InceptionTime":
        train_set = SignalDataset(step=12000, window_size=8500, bin_setup=train_config, source_dtype="float32")
    else:
        train_set = SignalDataset(step=10000, window_size=8500, bin_setup=train_config, source_dtype="float32")

    if network == "LSTM":
        nn_config["model"]["kwargs"]["layers"]["lstm_config"][0]["kwargs"]["input_size"] = config["input_size"]
    if network == "LSTM-CNN":
        print(nn_config["model"]["kwargs"]["layers"]["lstm_config"][0]["kwargs"]["input_size"])
        nn_config["model"]["kwargs"]["layers"]["lstm_config"][0]["kwargs"]["input_size"] = config["input_size"]
        nn_config["model"]["kwargs"]["layers"]["output_config"][0]["kwargs"]["in_features"] = (
                    ((config["input_size"] - 56) // 12) + 1)

    neuro_net = NeuroNet(config=nn_config, tensorboard=True)

    match network:
        case "InceptionTime":
            neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(), lr=nn_config["training_params"]["lr"])
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(neuro_net.optimizer,
                                                                     T_0=(1+nn_config["training_params"][
                                                                              "epoch_num"] // 2)))

        case "ResNet":
            neuro_net.optimizer = optim.Adam(neuro_net._model.parameters(), lr=nn_config["training_params"]["lr"])
            scheduler = (
                torch.optim.lr_scheduler.ExponentialLR(neuro_net.optimizer, gamma=0.8))

        case "LSTM":
            neuro_net.optimizer = optim.Adam(neuro_net._model.parameters(), lr=nn_config["training_params"]["lr"])
            scheduler = (
                torch.optim.lr_scheduler.PolynomialLR(neuro_net.optimizer, power=0.5,
                                                      total_iters=nn_config["training_params"]["epoch_num"]))



        case "LSTM-CNN":
            neuro_net.optimizer = optim.RMSprop(neuro_net._model.parameters(), lr=nn_config["training_params"]["lr"])
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(neuro_net.optimizer,
                                                                     T_0=(1+nn_config["training_params"][
                                                                              "epoch_num"] // 2)))



    train_dl, val_dl = neuro_net._train_val_dl_split(train_set, train_idx=None, val_idx=None)

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


config = {
    "lr": tune.uniform(1e-6, 1e-1),
    "optimizer": tune.choice(["adam", "adamw", "sgd", "rmsprop"]),
    "scheduler": tune.choice(["cosine", "polynomial", "exponential", "warmup"]),
}

config_length = {
    "input_size": tune.grid_search([1000, 2500, 5000, 8500, 10000, 12500])
}
nn_models = ["ResNet", "InceptionTime"]

for network in nn_models:

    hebo = HEBOSearch(metric="accuracy", mode="max")
    # hebo.restore("/home/petr/ray_results/train_network_2024-06-13_22-10-15/searcher-state-2024-06-13_22-10-15.pkl")
    iter = {"InceptionTime": 7, "ResNet": 20, "LSTM": 20, "CNN_spec": 20, "LSTM-CNN": 20}
    iter_min = {"InceptionTime": 2, "ResNet": 5, "LSTM": 5, "CNN_spec": 5, "LSTM-CNN": 5}
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
        name="LENGTH-" + network,
        resources_per_trial={"cpu": 4, "gpu": 1},
        num_samples=5,
        scheduler=asha_scheduler,
        config=config_length,
        storage_path="/mnt/home2/hparams_checkpoints/",
        verbose=1,
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(best_trial)
    print(f"Best trial accuracy: {best_trial.last_result['accuracy']}")
    print(f"Best trial validation loss: {best_trial.last_result['loss']}")

    print(f"Best trial config: {best_trial.config}")

# tensorboard --logdir /tmp/ray/session_2024-06-16_22-52-02_765634_29559/artifacts/2024-06-16_22-52-05/main_params-InceptionTime/driver_artifacts