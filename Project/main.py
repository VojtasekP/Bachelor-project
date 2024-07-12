from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from dataset.signal_dataset import SignalDataset
from signal_model import NeuroNet
import re
import torch
def load_yaml(config_path: Path):
    with config_path.open(mode="r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.SafeLoader)

def main(model, channel_train, channel_test):

    window_size = 30000

    # MODEL INIT

    sample_rate = 1562500


    signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"

    # TRAIN
    train_config = [{"label": (int(i.stem) - 1) // 4,
                     "channels": len(list(i.glob('*' + channel_train + '.bin'))),
                     "interval": [0, int(4 * sample_rate)],
                     "bin_path": list(i.glob('*' + channel_train + '.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    validation_config = [{"label": (int(i.stem) - 1) // 4,
                          "channels": len(list(i.glob('*' + channel_train + '.bin'))),
                          "interval": [int(4 * sample_rate), int(4.5 * sample_rate)],
                          "bin_path": list(i.glob('*' + channel_train + '.bin'))[0]}
                         for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    test_config = [{"label": (int(i.stem) - 1) // 4,
                    "channels": len(list(i.glob('*' + channel_test + '.bin'))),
                    "interval": [int(4.5 * sample_rate), int(5 * sample_rate)],
                    "bin_path": list(i.glob('*' + channel_test + '.bin'))[0]}
                   for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    model_path = Path("configs/nn_configs/" + model + ".yaml")
    nn_config = load_yaml(model_path)
    neuro_net = NeuroNet(config=nn_config, metrics=True)
    cutoff = (2000, sample_rate // 15)
    train_set = SignalDataset(step=5000, window_size=window_size, bin_setup=train_config, cutoff=cutoff, source_dtype="float32")
    val_set = SignalDataset(step=5000, window_size=window_size, bin_setup=validation_config,cutoff=cutoff, source_dtype="float32")
    test_set = SignalDataset(step=5000, window_size=window_size, bin_setup=test_config, cutoff=cutoff,source_dtype="float32")

    traindl = DataLoader(train_set, **nn_config["training_params"].get("dataloader_params", {}), shuffle=True,
                         pin_memory=True)

    valdl = DataLoader(val_set, **nn_config["eval_params"], pin_memory=True)

    # neuro_net._model.load_state_dict(torch.load("trained_models/" + channel_train + "/InceptionTime.pt"))
    # TRAIN
    neuro_net.train(traindl, valdl, patience=12)
    save_path = "trained_models/" + channel_train + "/" + model +"more.pt"
    # neuro_net._model = torch.load(save_path)
    # print(neuro_net.val_loss_list)
    # print(neuro_net.trainable_params, neuro_net.total_params, save_path)
    neuro_net.save(save_path)

    # EVALUATE ON TEST SET
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)
    outputs = np.empty((0,), dtype=np.float32)
    targets = np.empty((0,), dtype=np.longdouble).flatten()
    for i, (input, target) in enumerate(test_dataloader):
        input, target = input.numpy(), target.numpy()
        output = neuro_net.predict(input, argmax=True)
        outputs = np.concatenate((outputs, output), axis=0)
        targets = np.concatenate((targets, target), axis=0)
    cm_savepath = "/home/petr/Documents/graphs_for_BC/cm/"
    neuro_net.plot_confusion_matrix(outputs, targets)

if __name__ == '__main__':
    channel_1 = 'ch1'
    channel_2 = 'ch2'
    channel_3 = 'ch3'
    models = ["LSTM"]
    for model in models:
        for channel in [channel_1, channel_2, channel_3]:
            main(model, channel, channel)


# TODO: learn pycharm keybindings, etc.  ctrl, shift, alt, arrows; ctrl+alt+m/v
