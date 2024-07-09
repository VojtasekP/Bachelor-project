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
def train_network(config):
    sample_rate = 1562500
    signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"
    train_config = [{"label": (int(i.stem) - 1) // 4,
                     "channels": len(list(i.glob('*' + config["channel"] + '.bin'))),
                     "interval": [0, int(4.5 * sample_rate)],
                     "bin_path": list(i.glob('*' + config["channel"] + '.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
    train_set = SignalDataset(step=10000, window_size=config["input_size"], bin_setup=train_config, source_dtype="float32")
    # nn_config["training_params"]["warmups"] = config["warmups"]
    nn_config["training_params"]["lr"] = config["lr"]
    nn_config["training_params"]["dataloader_params"]["batch_size"] = config["batch_size"]
    lstm_hidden_size = nn_config["model"]["kwargs"]["layers"]["lstm_config"][0]["kwargs"]["hidden_size"]
    lstm_bidirectional = nn_config["model"]["kwargs"]["layers"]["lstm_config"][0]["kwargs"]["bidirectional"]

    update_layer_argument(nn_config, "lstm", arg="hidden_size", value=config["lstm"]["hidden_size"])
    update_layer_argument(nn_config, "lstm", arg="num_layers", value=config["lstm"]["num_layers"])
    update_layer_argument(nn_config, "lstm", arg="bidirectional", value=config["lstm"]["bidirectional"])


    lstm_output_size = lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size

    # Update LSTM input size in config
    update_layer_argument(nn_config, "lstm", "input_size", config["input_size"])

    update_layer_argument(nn_config, "conv1", "output_size", config["conv1"]["output_size"])
    update_layer_argument(nn_config, "conv1", "kernel_size", config["conv1"]["kernel_size"])
    update_layer_argument(nn_config, "conv1", "stride", config["conv1"]["stride"])
    update_layer_argument(nn_config, "conv1", "padding", config["conv1"]["padding"])


    update_layer_argument(nn_config, "conv2", "output_size", config["conv2"]["output_size"])
    update_layer_argument(nn_config, "conv2", "kernel_size", config["conv2"]["kernel_size"])
    update_layer_argument(nn_config, "conv2", "stride", config["conv2"]["stride"])
    update_layer_argument(nn_config, "conv2", "padding", config["conv2"]["padding"])


    update_layer_argument(nn_config, "conv3", "output_size", config["conv3"]["output_size"])
    update_layer_argument(nn_config, "conv3", "kernel_size", config["conv3"]["kernel_size"])
    update_layer_argument(nn_config, "conv3", "stride", config["conv3"]["stride"])
    update_layer_argument(nn_config, "conv3", "padding", config["conv3"]["padding"])

    # CNN layer configurations
    conv1_out_channels = nn_config["model"]["kwargs"]["layers"]["fcn_config"][0]["kwargs"]["out_channels"]
    conv2_out_channels = nn_config["model"]["kwargs"]["layers"]["fcn_config"][3]["kwargs"]["out_channels"]
    conv2_out_channels = nn_config["model"]["kwargs"]["layers"]["fcn_config"][5]["kwargs"]["out_channels"]

    # CNN layer parameters (example)
    kernel_size_conv1 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][0]["kwargs"]["kernel_size"]
    kernel_size_conv2 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][3]["kwargs"]["kernel_size"]
    kernel_size_conv3 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][6]["kwargs"]["kernel_size"]

    stride_conv1 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][0]["kwargs"]["stride"]
    stride_conv2 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][3]["kwargs"]["stride"]
    stride_conv3 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][6]["kwargs"]["stride"]

    padding_conv1 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][0]["kwargs"]["padding"]
    padding_conv2 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][3]["kwargs"]["padding"]
    padding_conv3 = nn_config["model"]["kwargs"]["layers"]["fcn_config"][6]["kwargs"]["padding"]
    # Calculate CNN output sizes (example, using the formula)
    def calculate_conv_output_size(input_size, kernel_size, stride, padding):
        return (input_size - kernel_size + 2 * padding) // stride + 1

    # Example computation
    input_size = 1000
    conv1_output_size = calculate_conv_output_size(input_size, kernel_size_conv1, stride=stride_conv1, padding=padding_conv1)
    conv2_output_size = calculate_conv_output_size(conv1_output_size, kernel_size_conv2, stride=stride_conv2, padding=padding_conv2)
    conv3_output_size = calculate_conv_output_size(conv2_output_size, kernel_size_conv3, stride=stride_conv2, padding=padding_conv2)

    update_layer_argument(nn_config, "adaptpool", arg="adaptpool", config["adaptpool"])

    update_layer_argument(nn_config, "linear1", arg="in_features", value=lstm_output_size+ conv_output)
    update_layer_argument(nn_config, "linear1", arg="out_features", value=config["linear1"])
    update_layer_argument(nn_config, "linear2", arg="in_features", value=config["linear1"])
    update_layer_argument(nn_config, "dropout", arg="p", value=config["dropout"])
    neuro_net = NeuroNet(config=nn_config, tensorboard=True)

    neuro_net.optimizer = optim.Adam(neuro_net._model.parameters(), lr=nn_config["training_params"]["lr"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(neuro_net.optimizer, T_0=(1+ nn_config["training_params"]["epoch_num"]//nn_config["training_params"]["warmups"])))

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



config = {
    "lstm": {
        "hidden_size": tune.randint(128,2500),
        "num_layers": tune.choice([1, 2, 3, 4])},
    "conv1": tune.choice([1, 2]),
    "conv2": tune.choice([1, 2]),
    "linear1": tune.randint(512,5000),
    "linear2": tune.randint(200, 3000),
    "lr": tune.uniform(1e-5, 1e-1),
    "dropout": tune.uniform(0, 0.5),
    "batch_size": tune.choice([128, 256, 512, 1024]),
    "epoch_num": tune.choice([5, 10, 15, 20, 40]),
}

network = "CNN_spec"
nn_config = load_yaml(Path("/home/petr/Documents/bachelor_project/Project/nn_configs/" + network + ".yaml"))
hebo = HEBOSearch(metric="accuracy", mode="max")
# hebo.restore("/home/petr/ray_results/train_network_2024-06-13_22-10-15/searcher-state-2024-06-13_22-10-15.pkl")

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
    partial(train_network),
    resources_per_trial={"cpu": 10, "gpu": 1},
    num_samples=200,
    checkpoint_config=CheckpointConfig(num_to_keep=5),
    search_alg=hebo,
    config=config,
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