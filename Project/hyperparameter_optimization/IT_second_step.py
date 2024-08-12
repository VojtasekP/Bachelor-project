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

# test_set = SignalDataset(step=10000, window_size=1000, bin_setup=test_config, source_dtype="float32")
def train_network(config):

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

    update_layer_argument(nn_config, "adaptpool1", "output_size", config["adaptivepool1"])
    update_layer_argument(nn_config, "inceptionblock1", arg="n_filters",
                          value=config["inceptionblock"]["n_filters"])

    update_layer_argument(nn_config, "inceptionblock1", arg="bottleneck_channels",
                          value=config["inceptionblock"]["bottleneck_channels"])


    update_layer_argument(nn_config, "inceptionblock2", arg="in_channels",
                          value=4*(config["inceptionblock"]["n_filters"]))

    update_layer_argument(nn_config, "inceptionblock2", arg="n_filters",
                          value=config["inceptionblock"]["n_filters"])

    update_layer_argument(nn_config, "inceptionblock2", arg="bottleneck_channels",
                          value=config["inceptionblock"]["bottleneck_channels"])


    update_layer_argument(nn_config, "adaptpool2", arg="output_size", value=config["adaptivepool2"])
    update_layer_argument(nn_config, "linear1", arg="in_features",
                          value=(4*config["inceptionblock"]["n_filters"]*config["adaptivepool2"]))
    update_layer_argument(nn_config, "linear1", arg="out_features", value=config["linear1"])
    update_layer_argument(nn_config, "linear2", arg="in_features", value=config["linear1"])


    neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(),
                                     **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
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
                     "lr": scheduler.get_last_lr()[0]})




loaded_signal = []
sample_rate = 1562500
channel = 'ch2'
signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"

config_second = {
    "adaptivepool1": tune.choice([400, 1000, 2000, 2500, 4000, 5000]),
    "inceptionblock": {
        "n_filters": tune.qrandint(2, 20, 2),
        "bottleneck_channels": tune.qrandint(2, 26, 2),
        "kernel_sizes": [tune.choice([2 * i + 1 for i in range(5, 23)]),
                          tune.choice([2 * i + 1 for i in range(25, 60)]),
                          tune.choice([2 * i + 1 for i in range(60, 100)])]},
    "adaptivepool2": tune.randint(5, 18),
    "linear1": tune.randint(32, 1000)
}

network = "InceptionTime"
nn_config = load_yaml(Path("../configs/nn_configs/" + network + ".yaml"))
hebo = HEBOSearch(metric="accuracy", mode="max")
hebo.restore("/mnt/home2/hparams_checkpoints/third_step_2_-IT/searcher-state-2024-07-03_08-22-10.pkl")

optuna_search = OptunaSearch(
    metric="accuracy",
    mode="max")

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='accuracy',
    mode='max',
    max_t=51,
    grace_period=15,
)
 # ②
result = tune.run(
    partial(train_network),
    resources_per_trial={"cpu": 10, "gpu": 1},
    name="third_step_2_-IT",
    num_samples=100,
    search_alg=hebo,
    storage_path="/mnt/home2/hparams_checkpoints/",
    # config=config_second,
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