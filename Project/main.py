import numpy as np
from torch.utils.data import DataLoader
from signal_model import NeuroNet, load_yaml
import re
from pathlib import Path
from dataset.signal_dataset import SignalDataset


def main():
    model = "LSTM"
    model_path = Path("configs/nn_configs/"+ model + ".yaml")
    nn_config = load_yaml(model_path)
    neuro_net = NeuroNet(config=nn_config, tensorboard=True)

    sample_rate = 1562500
    channel_1 = 'ch1'
    channel_2 = 'ch2'
    channel_3 = 'ch3'
    channel_train = channel_2
    channel_val = channel_2

    signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"


    train_config =[{"label": (int(i.stem) - 1) // 4,
                      "channels": len(list(i.glob('*' + channel_train + '.bin'))),
                      "interval": [0, int( 4.5* sample_rate)],
                      "bin_path": list(i.glob('*' + channel_train + '.bin'))[0]}
                     for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]


    test_config = [{"label": (int(i.stem) - 1) // 4,
                     "channels": len(list(i.glob('*' + channel_val + '.bin'))),
                     "interval": [int(4.5 * sample_rate), int(5 * sample_rate)],
                     "bin_path": list(i.glob('*' + channel_val + '.bin'))[0]}
                    for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

    train_set = SignalDataset(step=5000, window_size=8500, bin_setup=train_config, source_dtype="float32")
    test_set = SignalDataset(step=5000, window_size=8500, bin_setup=test_config, source_dtype="float32")
    neuro_net.train(train_set)
    save_path = "trained_models/" + channel_train + "/" + model +".pt"
    # neuro_net._model = torch.load(save_path)
    # print(neuro_net.val_loss_list)
    print(neuro_net.trainable_params, neuro_net.total_params, save_path)
    neuro_net.save(save_path)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)
    outputs = np.empty((0,), dtype=np.float32)
    targets = np.empty((0,), dtype=np.longdouble).flatten()
    for i, (input, target) in enumerate(test_dataloader):
        input, target = input.numpy(), target.numpy()
        output = neuro_net.predict(input, argmax=True)
        outputs = np.concatenate((outputs, output), axis=0)
        targets = np.concatenate((targets, target), axis=0)
    neuro_net.plot_confusion_matrix(outputs, targets)


if __name__ == '__main__':
    main()

# TODO: learn pycharm keybindings, etc.  ctrl, shift, alt, arrows; ctrl+alt+m/v
