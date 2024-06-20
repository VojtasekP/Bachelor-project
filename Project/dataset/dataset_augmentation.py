import numpy as np
from torch.utils.data import random_split, DataLoader
from signal_model import NeuroNet, SklearnModel, load_yaml
import re
from pathlib import Path
from signal_dataset import SignalDataset

nn_config = load_yaml(Path("../configs/nn_configs/CNN_spec.yaml"))
neuro_net = NeuroNet(config=nn_config, tensorboard=True)

sample_rate = 1562500
channel_1 = 'ch1'
channel_2 = 'ch2'
channel_3 = 'ch3'
signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"
train_config = ([{"label": (int(i.stem) - 1) // 4,
                  "channels": len(list(i.glob('*' + channel_1 + '.bin'))),
                  "interval": [0, int(4.5 * sample_rate)],
                  "bin_path": list(i.glob('*' + channel_1 + '.bin'))[0]}
                 for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
                +
                [{"label": (int(i.stem) - 1) // 4,
                  "channels": len(list(i.glob('*' + channel_2 + '.bin'))),
                  "interval": [0, int(4.5 * sample_rate)],
                  "bin_path": list(i.glob('*' + channel_2 + '.bin'))[0]}
                 for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
                +
                [{"label": (int(i.stem) - 1) // 4,
                  "channels": len(list(i.glob('*' + channel_3 + '.bin'))),
                  "interval": [0, int(4.5 * sample_rate)],
                  "bin_path": list(i.glob('*' + channel_3 + '.bin'))[0]}
                 for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)])

test_config = ([{"label": (int(i.stem) - 1) // 4,
                 "channels": len(list(i.glob('*' + channel_1 + '.bin'))),
                 "interval": [int(4.5 * sample_rate), int(5 * sample_rate)],
                 "bin_path": list(i.glob('*' + channel_1 + '.bin'))[0]}
                for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
               +
               [{"label": (int(i.stem) - 1) // 4,
                 "channels": len(list(i.glob('*' + channel_2 + '.bin'))),
                 "interval": [int(4.5 * sample_rate), int(5 * sample_rate)],
                 "bin_path": list(i.glob('*' + channel_2 + '.bin'))[0]}
                for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]
               +
               [{"label": (int(i.stem) - 1) // 4,
                 "channels": len(list(i.glob('*' + channel_3 + '.bin'))),
                 "interval": [int(4.5 * sample_rate), int(5 * sample_rate)],
                 "bin_path": list(i.glob('*' + channel_3 + '.bin'))[0]}
                for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)])

train_set = SignalDataset(step=1000, window_size=1000, bin_setup=train_config, source_dtype="float32")
test_set = SignalDataset(step=1000, window_size=1000, bin_setup=test_config, source_dtype="float32")

neuro_net.train(train_set)
save_path = "../trained_models/ch2/LSTM_CNN.pt"
neuro_net.save(save_path)
test_dataloader = DataLoader(test_set, batch_size=512, shuffle=False)
outputs = np.empty((0,), dtype=np.float32)
targets = np.empty((0,), dtype=np.longdouble).flatten()
for i, (input, target) in enumerate(test_dataloader):
    input, target = input.numpy(), target.numpy()
    output = neuro_net.predict(input)
    outputs = np.concatenate((outputs, output), axis=0)
    targets = np.concatenate((targets, target), axis=0)
neuro_net.plot_confusion_matrix(outputs, targets)