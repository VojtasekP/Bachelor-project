import re

import torch

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
def train_network(config, network):

    sample_rate = 1562500
    signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"

    train_config = [{"label": (int(i.stem) - 1) // 4,
                     "channels": len(list(i.glob('*ch2.bin'))),
                     "interval": [0, int(4.5 * sample_rate)],
                     "bin_path": list(i.glob('*ch2.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    train_set = SignalDataset(step=config["step"], window_size=8500, bin_setup=train_config, source_dtype="float32")
    length = len(train_set)
    if network == "LSTM":
        nn_config["model"]["kwargs"]["layers"]["lstm_config"][0]["kwargs"]["input_size"] = config["input_size"]

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
                     "lr": scheduler.get_last_lr()[0], "train_samples": length})
    # with tempfile.TemporaryDirectory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "wb") as fp:
    #         pickle.dump(checkpoint_data, fp)
    #     checkpoint = Checkpoint.from_directory(checkpoint_dir)
    #     train.report({"loss": neuro_net.val_loss, "accuracy": neuro_net.val_accuracy[-1]}, checkpoint=checkpoint)



config = {
    "step": tune.grid_search([5000, 7500, 10000, 12500, 15000, 25000, 50000])
}
nn_models = ["InceptionTime"]

for network in nn_models:

    nn_config = load_yaml(Path("/home/petr/Documents/bachelor_project/Project/nn_configs/" + network + ".yaml"))
    hebo = HEBOSearch(metric="accuracy", mode="max")
    # hebo.restore("/home/petr/ray_results/train_network_2024-06-13_22-10-15/searcher-state-2024-06-13_22-10-15.pkl")

    optuna_search = OptunaSearch(
        metric="accuracy",
        mode="max")


    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='accuracy',
        mode='max',
        max_t=20,
        grace_period=5,
    )

     # ②
    result = tune.run(
        partial(train_network, network=network),
        name="num_of_samples-" + network,
        resources_per_trial={"cpu": 10, "gpu": 1},
        num_samples=3,
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

# │ adaptivepool                                 10 │
# │ batch_size                                  256 │
# │ inceptionblock1/bottleneck_channels           4 │
# │ inceptionblock1/n_filters                    15 │
# │ inceptionblock2/bottleneck_channels          47 │
# │ inceptionblock2/n_filters                    20 │
# │ linear1                                     305 │
# │ lr                                      0.00251