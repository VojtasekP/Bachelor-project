import re

import torch

from ray.air import CheckpointConfig
from ray.tune.search.hebo import HEBOSearch
from torch import optim
from dataset.signal_dataset import SignalDataset
from ray.tune.search.optuna import OptunaSearch
from ray import train, tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from signal_model import NeuroNet
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
def calculate_conv1d_output_size(input_size, kernel_size, stride, padding):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

def calculate_pool1d_output_size(input_size, kernel_size):
    return ((input_size - kernel_size) // kernel_size) + 1
# test_set = SignalDataset(step=10000, window_size=1000, bin_setup=test_config, source_dtype="float32")
def train_network(config, network):
    sample_rate = 1562500
    signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"
    train_config = [{"label": (int(i.stem) - 1) // 4,
                     "channels": len(list(i.glob('*ch2.bin'))),
                     "interval": [0, int(4.5 * sample_rate)],
                     "bin_path": list(i.glob('*ch2.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
    nn_config = load_yaml(Path("/home/petr/Documents/bachelor_project/Project/configs/nn_configs/" + network + ".yaml"))

    train_set = SignalDataset(step=10000, window_size=12000, bin_setup=train_config, source_dtype="float32")


    update_layer_argument(nn_config, "lstm", arg="hidden_size", value=config["lstm"]["hidden_size"])
    update_layer_argument(nn_config, "lstm", arg="num_layers", value=config["lstm"]["num_layers"])
    if config["lstm"]["num_layers"]>1:
        update_layer_argument(nn_config, "lstm", arg="dropout", value=config["lstm"]["lstm_dropout"])
    if config["lstm"]["bidirectional"]:
        update_layer_argument(nn_config, "lstm", arg="bidirectional", value=True)
        update_layer_argument(nn_config, "linear1", arg="in_features", value=config["lstm"]["hidden_size"]*2)
        lstm_output = 2*config["lstm"]["hidden_size"]
    else:
        update_layer_argument(nn_config, "lstm", arg="bidirectional", value=False)
        update_layer_argument(nn_config, "linear1", arg="in_features", value=config["lstm"]["hidden_size"])
        lstm_output = config["lstm"]["hidden_size"]


    update_layer_argument(nn_config, "conv1", "out_channels", config["conv1"]["out_channels"])
    update_layer_argument(nn_config, "conv1", "kernel_size", config["conv1"]["kernel_size"])
    update_layer_argument(nn_config, "conv1", "stride", config["conv1"]["stride"])
    update_layer_argument(nn_config, "conv1", "padding", config["conv1"]["padding"])
    update_layer_argument(nn_config, "bn1", "num_features", config["conv1"]["out_channels"])

    conv1_output_size = calculate_conv1d_output_size(12000, config["conv1"]["kernel_size"], config["conv1"]["stride"], config["conv1"]["padding"])

    update_layer_argument(nn_config, "conv2", "in_channels", config["conv1"]["out_channels"])
    update_layer_argument(nn_config, "conv2", "out_channels", config["conv2"]["out_channels"])
    update_layer_argument(nn_config, "conv2", "kernel_size", config["conv2"]["kernel_size"])
    update_layer_argument(nn_config, "conv2", "stride", config["conv2"]["stride"])
    update_layer_argument(nn_config, "conv2", "padding", config["conv2"]["padding"])
    update_layer_argument(nn_config, "bn2", "num_features", config["conv2"]["out_channels"])

    conv2_output_size = calculate_conv1d_output_size(conv1_output_size, config["conv2"]["kernel_size"], config["conv2"]["stride"], config["conv2"]["padding"])

    update_layer_argument(nn_config, "conv3", "in_channels", config["conv2"]["out_channels"])
    update_layer_argument(nn_config, "conv3", "out_channels", config["conv3"]["out_channels"])
    update_layer_argument(nn_config, "conv3", "kernel_size", config["conv3"]["kernel_size"])
    update_layer_argument(nn_config, "conv3", "stride", config["conv3"]["stride"])
    update_layer_argument(nn_config, "conv3", "padding", config["conv3"]["padding"])
    update_layer_argument(nn_config, "bn3", "num_features", config["conv3"]["out_channels"])

    conv3_output_size = calculate_conv1d_output_size(conv2_output_size, config["conv3"]["kernel_size"], config["conv3"]["stride"], config["conv3"]["padding"])

    update_layer_argument(nn_config, "avgpool", arg="kernel_size", value=config["avgpool"]["kernel_size"])
    conv_output = config["conv3"]["out_channels"] * calculate_pool1d_output_size(conv3_output_size, config["avgpool"]["kernel_size"])


    update_layer_argument(nn_config, "linear1", arg="in_features", value=lstm_output+conv_output)
    update_layer_argument(nn_config, "linear1", arg="out_features", value=config["linear1"])
    update_layer_argument(nn_config, "linear2", arg="in_features", value=config["linear1"])
    update_layer_argument(nn_config, "dropout", arg="p", value=config["dropout"])
    # print(nn_config)
    neuro_net = NeuroNet(config=nn_config, tensorboard=True)

    neuro_net.optimizer = optim.Adam(neuro_net._model.parameters(), lr=neuro_net.config["training_params"]['optimizer_params']["kwargs"]["lr"])
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(neuro_net.optimizer, **nn_config["training_params"]["scheduler_params"].get("kwargs", {})))

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
    # with tempfile.TemporaryDirectory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "wb") as fp:
    #         pickle.dump(checkpoint_data, fp)
    #     checkpoint = Checkpoint.from_directory(checkpoint_dir)
    #     train.report({"loss": neuro_net.val_loss, "accuracy": neuro_net.val_accuracy[-1]}, checkpoint=checkpoint)



config = {
    "lstm": {
        "hidden_size": tune.randint(256,2000),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "bidirectional": tune.choice([True, False]),
        "lstm_dropout": tune.uniform(0.1, 0.5),},
    "conv1": {
        "out_channels":tune.randint(1,12),
        "kernel_size": tune.randint(5,40),
        "stride": tune.randint(1,10),
        "padding": tune.randint(1,5),
    },
    "conv2": {
        "out_channels":tune.randint(8,18),
        "kernel_size": tune.randint(5,40),
        "stride": tune.randint(1,10),
        "padding": tune.randint(1,5),
    },
    "conv3": {
        "out_channels": tune.randint(14, 24),
        "kernel_size": tune.randint(5, 40),
        "stride": tune.randint(1, 10),
        "padding": tune.randint(1, 5),
    },
    "avgpool": {"kernel_size":tune.randint(10, 40),
                "stride": tune.randint(2,15)},
    "linear1": tune.randint(256,2000),
    "linear2": tune.randint(128,1000),
    "dropout": tune.uniform(0, 0.5)
}

nn_models = ["LSTM-CNN"]

for network in nn_models:

    hebo = HEBOSearch(metric="accuracy", mode="max")
    # hebo.restore("/home/petr/ray_results/train_network_2024-06-13_22-10-15/searcher-state-2024-06-13_22-10-15.pkl")
    iter = {"InceptionTime": 17, "ResNet": 16, "LSTM": 17, "CNN_spec": 20, "LSTM-CNN": 13}
    iter_min = {"InceptionTime": 5, "ResNet": 8, "LSTM": 5, "CNN_spec": 5, "LSTM-CNN": 5}
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
        name="THIRD_STEP_" + network,
        resources_per_trial={"cpu": 10, "gpu": 1},
        num_samples=100,
        scheduler=asha_scheduler,
        search_alg=hebo,
        config=config,
        storage_path="/mnt/home2/hparams_checkpoints/",
        verbose=1,
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(best_trial)
    print(f"Best trial accuracy: {best_trial.last_result['accuracy']}")
    print(f"Best trial validation loss: {best_trial.last_result['loss']}")

    print(f"Best trial config: {best_trial.config}")