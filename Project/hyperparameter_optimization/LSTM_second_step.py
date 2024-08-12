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
    channel = 'ch2'
    signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"
    train_config = [{"label": (int(i.stem) - 1) // 4,
                     "channels": len(list(i.glob('*' + channel + '.bin'))),
                     "interval": [0, int(4 * sample_rate)],
                     "bin_path": list(i.glob('*' + channel + '.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
    val_config = [{"label": (int(i.stem) - 1) // 4,
                   "channels": len(list(i.glob('*' + channel + '.bin'))),
                   "interval": [int(4 * sample_rate), int(4.5 * sample_rate)],
                   "bin_path": list(i.glob('*' + channel + '.bin'))[0]}
                  for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
    train_set = SignalDataset(step=10000, window_size=5000, bin_setup=train_config, source_dtype="float32")
    val_set = SignalDataset(step=10000, window_size=5000, bin_setup=val_config, source_dtype="float32")

    # update_layer_argument(nn_config, "adaptivepool", "output_size", value=config["adaptivepool"])
    update_layer_argument(nn_config, "lstm", "input_size", value=config["lstm"]["input_size"])
    update_layer_argument(nn_config, "lstm", arg="hidden_size", value=config["lstm"]["hidden_size"])
    update_layer_argument(nn_config, "lstm", arg="num_layers", value=config["lstm"]["num_layers"])

    update_layer_argument(nn_config, "lstm", arg="dropout", value=config["lstm"]["lstm_dropout"])
    if config["lstm"]["bidirectional"]:
        update_layer_argument(nn_config, "lstm", arg="bidirectional", value=True)
        update_layer_argument(nn_config, "linear1", arg="in_features", value=config["lstm"]["hidden_size"]*2)

    else:
        update_layer_argument(nn_config, "lstm", arg="bidirectional", value=False)
        update_layer_argument(nn_config, "linear1", arg="in_features", value=config["lstm"]["hidden_size"])

    update_layer_argument(nn_config, "linear1", arg="out_features", value=config["linear1"])
    update_layer_argument(nn_config, "linear2", arg="in_features", value=int(
        config["linear1"] * (5000 // config["lstm"]["input_size"])))
    update_layer_argument(nn_config, "dropout", arg="p", value=config["dropout"])

    neuro_net = NeuroNet(config=nn_config, metrics=True)

    neuro_net.optimizer = optim.AdamW(neuro_net._model.parameters(),
                                     **neuro_net.config["training_params"]["optimizer_params"].get("kwargs", {}))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neuro_net.optimizer,
                                                           T_max=nn_config["training_params"]["epoch_num"])

    train_dl = DataLoader(train_set, **nn_config["training_params"].get("dataloader_params", {}), shuffle=True,
                         pin_memory=True)

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


config = {
    # "adaptivepool": tune.choice([500, 1000, 1250, 2000, 2500, ]),
    "lstm": {
        "input_size": tune.choice([10, 20, 25, 40, 50, 100, 125, 200]),
        "hidden_size": tune.randint(20,200),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "bidirectional": tune.choice([True, False]),
        "lstm_dropout": tune.uniform(0, 0.5)},
    "linear1": tune.randint(2,25),
    "dropout": tune.uniform(0, 0.5),
}


nn_config = load_yaml(Path("../configs/nn_configs/LSTM.yaml"))
hebo = HEBOSearch(metric="loss", mode="min")

optuna_search = OptunaSearch(
    metric="loss",
    mode="min")

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='loss',
    mode='min',
    max_t=40,
    grace_period=10,
)
 # ②
result = tune.run(
    partial(train_network),
    resources_per_trial={"cpu": 10, "gpu": 1},
    name="SECSTEP-LSTM",
    num_samples=150,
    search_alg=hebo,
    storage_path="/mnt/home2/hparams_checkpoints/",
    config=config,
    scheduler=asha_scheduler,
    verbose=1,
)


best_trial = result.get_best_trial("accuracy", "max", "last")
print(best_trial)
print(f"Best trial accuracy: {best_trial.last_result['accuracy']}")
print(f"Best trial validation loss: {best_trial.last_result['loss']}")

print(f"Best trial config: {best_trial.config}")
