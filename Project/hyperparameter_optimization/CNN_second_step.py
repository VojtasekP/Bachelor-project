import math
import re

import torch

from ray.air import CheckpointConfig
from ray.tune.search.hebo import HEBOSearch
from torch import optim
from torch.utils.data import DataLoader

from dataset.signal_dataset import SignalDataset
from ray.tune.search.optuna import OptunaSearch
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
def calculate_conv_output_size(input_size, kernel_size, padding, dilation, stride = 1):
    output_size = math.floor(
        (input_size - dilation * (kernel_size - 1) - 1 + 2 * padding) / stride
    ) + 1
    return output_size
# test_set = SignalDataset(step=10000, window_size=1000, bin_setup=test_config, source_dtype="float32")
def train_cnn(config):
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

    train_set = SignalDataset(step=10000, window_size=5000, bin_setup=train_config, source_dtype="float32")
    val_set = SignalDataset(step=10000, window_size=5000, bin_setup=val_config, source_dtype="float32")
    update_layer_argument(nn_config, 'conv1', 'out_channels', config['first_conv']["out_channels"])
    update_layer_argument(nn_config, 'conv1', 'kernel_size', config['first_conv']['kernel_size'])
    update_layer_argument(nn_config, 'conv1', 'padding', config['first_conv']['kernel_size'] // 2)
    update_layer_argument(nn_config, 'conv1', 'dilation', config['dilation'])
    update_layer_argument(nn_config, 'avgpool1', 'kernel_size', config['pool'])
    output_size = calculate_conv_output_size(input_size=5000, kernel_size=config['first_conv']['kernel_size'], dilation=config['dilation'], padding=config['first_conv']['kernel_size']//2)
    print(output_size)
    update_layer_argument(nn_config, 'bn1', 'num_features', config['first_conv']["out_channels"])
    #
    update_layer_argument(nn_config, 'conv2', 'in_channels', config['first_conv']["out_channels"])
    update_layer_argument(nn_config, 'conv2', 'out_channels', config['second_conv']["out_channels"])
    update_layer_argument(nn_config, 'conv2', 'kernel_size', config['second_conv']['kernel_size'])
    update_layer_argument(nn_config, 'conv2', 'padding', config['second_conv']['kernel_size'] // 2)
    update_layer_argument(nn_config, 'conv2', 'dilation', config['dilation'])
    update_layer_argument(nn_config, 'avgpool2', 'kernel_size', config['pool'])
    output_size = calculate_conv_output_size(input_size=output_size, kernel_size=config['second_conv']['kernel_size'], dilation=config['dilation'], padding=config['second_conv']['kernel_size']//2)
    print(output_size)
    #
    update_layer_argument(nn_config, 'bn2', 'num_features', config['second_conv']["out_channels"])
    #
    update_layer_argument(nn_config, 'conv3', 'in_channels', config['second_conv']["out_channels"])
    update_layer_argument(nn_config, 'conv3', 'out_channels', config['third_conv']["out_channels"])
    update_layer_argument(nn_config, 'conv3', 'kernel_size', config['third_conv']['kernel_size'])
    update_layer_argument(nn_config, 'conv3', 'padding', config['third_conv']['kernel_size'] // 2)
    update_layer_argument(nn_config, 'conv3', 'dilation', config['dilation'])
    update_layer_argument(nn_config, 'avgpool3', 'kernel_size', config['pool'])
    #
    output_size = calculate_conv_output_size(input_size=output_size, kernel_size=config['third_conv']['kernel_size'], dilation=config['dilation'], padding=config['third_conv']['kernel_size']//2)
    print(output_size)
    update_layer_argument(nn_config, 'bn3', 'num_features', config['third_conv']["out_channels"])
    #
    update_layer_argument(nn_config, 'conv4', 'in_channels', config['third_conv']["out_channels"])
    update_layer_argument(nn_config, 'conv4', 'out_channels', config['fourth_conv']["out_channels"])
    update_layer_argument(nn_config, 'conv4', 'kernel_size', config['fourth_conv']['kernel_size'])
    update_layer_argument(nn_config, 'conv4', 'padding', config['fourth_conv']['kernel_size'] // 2)
    update_layer_argument(nn_config, 'avgpool4', 'kernel_size', config['pool'])
    #
    update_layer_argument(nn_config, 'bn4', 'num_features', config['fourth_conv']["out_channels"])
    #
    output_size = calculate_conv_output_size(input_size=output_size, kernel_size=config['fourth_conv']['kernel_size'], dilation=config['dilation'], padding=config['fourth_conv']['kernel_size']//2)
    print(output_size)
    update_layer_argument(nn_config, 'linear1', 'in_features', output_size*int(config['fourth_conv']["out_channels"]*config['pool']))
    update_layer_argument(nn_config, 'linear1', 'out_features', config['linear'])
    update_layer_argument(nn_config, 'dropout', 'p', config['dropout'])
    update_layer_argument(nn_config, 'linear2', 'in_features', config['linear'])
    neuro_net = NeuroNet(config=nn_config, metrics=True)
    print(nn_config)
    neuro_net.optimizer = optim.Adam(neuro_net._model.parameters(),
                                     **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
                                                           T_max=nn_config["training_params"]["epoch_num"])


    start_epoch = 0
    running_loss = 0.0
    train_dl = DataLoader(train_set, **nn_config["training_params"].get("dataloader_params", {}), shuffle=True,
                         pin_memory=True)

    val_dl = DataLoader(val_set, **nn_config["eval_params"], pin_memory=True)
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


config = {
    "dilation": tune.randint(1, 5),
    "first_conv" :{
        "out_channels": tune.qrandint(2,32, 2),
        "kernel_size": tune.randint(2,80),
    },
    "second_conv": {
        "out_channels": tune.qrandint(4,64, 2),
        "kernel_size": tune.randint(2, 80),
    },
    "third_conv": {
        "out_channels": tune.qrandint(8,64, 2),
        "kernel_size": tune.randint(2, 80),
    },
    "fourth_conv": {
        "out_channels": tune.qrandint(8,64, 2),
        "kernel_size": tune.randint(2, 80),
    },
    "pool": tune.randint(1, 5),
    "dropout": tune.uniform(0, 0.5),
    "linear": tune.randint(5, 1000),
}



network = "CNN"
nn_config = load_yaml(Path("/home/petr/Documents/bachelor_project/Project/configs/nn_configs/" + network + ".yaml"))
hebo = HEBOSearch(metric="loss", mode="min")

optuna_search = OptunaSearch(
    metric="accuracy",
    mode="max")

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='loss',
    mode='min',
    max_t=50,
    grace_period=10,
)
 # ②
result = tune.run(
    partial(train_cnn),
    name="SECSTEP-CNN",
    resources_per_trial={"cpu": 10, "gpu": 1},
    num_samples=150,
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

