from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from dataset.signal_dataset import SignalDataset
from dataset.load_yaml import load_yaml
from signal_model import NeuroNet
import re
import torch


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
                          "channels": len(list(i.glob('*' + channel_test + '.bin'))),
                          "interval": [int(4 * sample_rate), int(4.5 * sample_rate)],
                          "bin_path": list(i.glob('*' + channel_test + '.bin'))[0]}
                         for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    test_config = [{"label": (int(i.stem) - 1) // 4,
                    "channels": len(list(i.glob('*' + channel_test + '.bin'))),
                    "interval": [int(4.5 * sample_rate), int(5 * sample_rate)],
                    "bin_path": list(i.glob('*' + channel_test + '.bin'))[0]}
                   for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]



    model_path = Path("configs/nn_configs/" + model + ".yaml")
    nn_config = load_yaml(model_path)
    neuro_net = NeuroNet(config=nn_config, metrics=True)
    if model == "InceptionTime":
        cutoff = (1000, sample_rate // 60)
    else:
        cutoff = (1000, sample_rate // 24)

    train_set = SignalDataset(step=5000, window_size=window_size, bin_setup=train_config,  cutoff=cutoff,source_dtype="float32")
    val_set = SignalDataset(step=5000, window_size=window_size, bin_setup=validation_config, cutoff=cutoff, source_dtype="float32")
    test_set = SignalDataset(step=5000, window_size=window_size, bin_setup=test_config, cutoff=cutoff, source_dtype="float32")

    traindl = DataLoader(train_set, **nn_config["training_params"].get("dataloader_params", {}), shuffle=True)

    valdl = DataLoader(val_set, **nn_config["eval_params"])

    # neuro_net._model.load_state_dict(torch.load("trained_models/" + channel_train + "/InceptionTime.pt"))
    # TRAIN
    neuro_net.train(traindl, valdl, patience=12)


    save_path = "trained_models/" + channel_train + "/" + model +" +_cross_channnel.pt"
    neuro_net.save(save_path)
    neuro_net._model.load_state_dict(torch.load((save_path)))
    print(neuro_net.trainable_params, neuro_net.total_params, save_path)


    # EVALUATE ON TEST SET
    test_dataloader = DataLoader(test_set, **nn_config["eval_params"])
    outputs = np.empty((0,), dtype=np.float32)
    targets = np.empty((0,), dtype=np.longdouble).flatten()
    for i, (input, target) in enumerate(test_dataloader):
        input, target = input.numpy(), target.numpy()
        output = neuro_net.predict(input, argmax=True)
        outputs = np.concatenate((outputs, output), axis=0)
        targets = np.concatenate((targets, target), axis=0)

    neuro_net.plot_confusion_matrix(outputs, targets)

if __name__ == '__main__':

    models = ["InceptionTime", "CNN", "LSTM"]
    channels = ['ch1', 'ch2', 'ch3']

    for model in models:
        for channel in channels:
            main(model, channel, channel)

