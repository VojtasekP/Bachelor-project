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

def calculate_output_size(input_size):
    conv1_output_size = (input_size - 30) + 1
    avgpool1_output_size = ((conv1_output_size - 5) // 5) + 1
    conv2_output_size = (avgpool1_output_size - 24) + 1
    avgpool2_output_size = ((conv2_output_size - 5) // 5) + 1
    conv3_output_size = (avgpool2_output_size - 16) + 1
    avgpool3_output_size = ((conv3_output_size - 5) // 5) + 1
    conv4_output_size = (avgpool3_output_size - 11) + 1
    avgpool4_output_size = ((conv4_output_size - 15) // 15) + 1
    return avgpool4_output_size

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

    # if network == "LSTM":
    #
    #     nn_config["model"]["kwargs"]["layers"]["lstm_config"][1]["kwargs"]["input_size"] = config["input_size"]
    # if network == "LSTM-CNN":


        # nn_config["model"]["kwargs"]["layers"]["lstm_config"][0]["kwargs"]["input_size"] = config["input_size"]

        # nn_config["model"]["kwargs"]["layers"]["output_config"][0]["kwargs"]["in_features"] = calculate_output_size(config["input_size"]) + 1024

    neuro_net = NeuroNet(config=nn_config, metrics=True)


    if network == "InceptionTime":
        match config["input_size"]:
            case 2500:
                cutoff_freq = sample_rate//2
            case 5000:
                cutoff_freq = sample_rate // 10
            case 10000:
                cutoff_freq = sample_rate // 20
            case 15000:
                cutoff_freq = sample_rate // 30
            case 20000:
                cutoff_freq = sample_rate // 40
            case 25000:
                cutoff_freq = sample_rate // 50
            case 30000:
                cutoff_freq = sample_rate // 60

    else:
        match config["input_size"]:
            case 2500:
                cutoff_freq = sample_rate//2
            case 5000:
                cutoff_freq = sample_rate // 2
            case 10000:
                cutoff_freq = sample_rate // 4
            case 15000:
                cutoff_freq = sample_rate // 6
            case 20000:
                cutoff_freq = sample_rate // 8
            case 25000:
                cutoff_freq = sample_rate // 10
            case 30000:
                cutoff_freq = sample_rate // 12

    neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(),
                                      **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
                                                           T_max=nn_config["training_params"]["epoch_num"])

        #     # cutoff = {2500: sample_rate // 2, 5000: sample_rate//3, 10000: sample_rate//5, 15000:sample_rate//8, 20000: sample_rate//10, 25000: sample_rate//13, 30000: sample_rate//15}
        #
        # case "LSTM":
        #     neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(), **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
        #                                                T_max=nn_config["training_params"]["epoch_num"])
        #     # cutoff = {2500: sample_rate , 5000: sample_rate//2, 10000: sample_rate//4, 15000:sample_rate//6, 20000: sample_rate//8, 25000: sample_rate//10, 30000: sample_rate//12}
        # case "CNN":
        #
        #     neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(), **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
        #                                                T_max=nn_config["training_params"]["epoch_num"])

            # cutoff = {2500: sample_rate, 5000: sample_rate, 10000: sample_rate, 15000:sample_rate, 20000: sample_rate, 25000: sample_rate, 30000: sample_rate}
    print(cutoff_freq)
    train_set = SignalDataset(step=10000, window_size=config["input_size"], bin_setup=train_config, cutoff=(1000,cutoff_freq), source_dtype="float32")
    val_set = SignalDataset(step=10000, window_size=config["input_size"], bin_setup=val_config, cutoff=(1000,cutoff_freq), source_dtype="float32")
    train_dl = DataLoader(train_set, **nn_config["training_params"].get("dataloader_params", {}), shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_set, **nn_config["eval_params"], pin_memory=True)

    start_epoch = 0
    running_loss = 0.0
    best_acc = 0
    early_stopper = EarlyStopper(patience=12)
    for epoch in range(start_epoch, nn_config["training_params"]["epoch_num"]):
        neuro_net.train_one_epoch(train_dl, running_loss)
        neuro_net.validate(val_dl)
        scheduler.step()

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": neuro_net._model.state_dict(),
            "optimizer_state_dict": neuro_net.optimizer.state_dict(),
        }
        acc = neuro_net.val_accuracy[-1]
        if acc > best_acc:
            best_acc = acc
        train.report({"loss": neuro_net.best_loss,
                      "accuracy": acc,
                      "lr": scheduler.get_last_lr()[0],
                      "epoch_trained": epoch,
                      "best_acc": best_acc})

        if early_stopper.early_stop(neuro_net.val_loss):
            print(f"Training stopped due to early stopping. Last epoch: {epoch}")
            break



config_length = {
    "input_size": tune.grid_search([2500, 5000, 10000, 15000, 20000, 25000, 30000])
}

cutoffs = {
     "cutoff": tune.grid_search([50000, 60000, 80000, 100000, 120000, 140000, 200000, 250000, 300000, 781250])
}

nn_models = ["InceptionTime", "LSTM", "CNN"]

for network in nn_models:

    hebo = HEBOSearch(metric="loss", mode="min")
    # hebo.restore("/home/petr/ray_results/train_network_2024-06-13_22-10-15/searcher-state-2024-06-13_22-10-15.pkl")
    iter = {"InceptionTime": 51, "LSTM": 51, "CNN": 51}
    iter_min = {"InceptionTime": 15,  "LSTM": 15, "CNN": 15}
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='accuracy',
        mode='max',
        max_t=51,
        grace_period=15,
    )

    # ②
    result = tune.run(
        partial(train_network, network=network),
        name="STEP_" + network,
        resources_per_trial={"cpu": 10, "gpu": 1},
        num_samples=5,
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