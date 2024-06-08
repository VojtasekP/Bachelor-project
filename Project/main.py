from torch.utils.data import random_split, DataLoader
from signal_model import NeuroNet, SklearnModel
import re
from pathlib import Path
from signal_dataset import SignalDataset


def main():
    neuro_net = NeuroNet(Path("nn_yaml_configs/MLP.yaml"), tensorboard=True)
    sr = 1562500
    signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"
    bin_setup = [{"label": int(i.stem)//4,
                  "channels": len(list(i.glob('*.bin'))),
                  "interval": [0, 5 * sr],
                  "bin_path": list(i.glob('*.bin'))[0]}
                 for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    print(bin_setup)
    sd = SignalDataset(step=1000, window_size=1000, bin_setup=bin_setup, source_dtype="float32")
    train_data, test_data = random_split(sd, [0.8, 0.2])

    neuro_net.train(train_data, test_data)
    neuro_net.close_writer()


if __name__ == '__main__':
    main()

# TODO: learn pycharm keybindings, etc.  ctrl, shift, alt, arrows; ctrl+alt+m/v
