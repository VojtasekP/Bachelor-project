from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from signal_model import NeuroNet
import re
from pathlib import Path
from dataset.signal_dataset import SignalDataset
import torch
import yaml

def load_yaml(config_path: Path):
    with config_path.open(mode="r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.SafeLoader)


def evaluate_models(models, channels):
    for model in models:
        for channel in channels:

            signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"



            test_config = [{"label": (int(i.stem) - 1) // 4,
                            "channels": len(list(i.glob('*' + channel + '.bin'))),
                            "interval": [int(4.5 * sample_rate), int(5 * sample_rate)],
                            "bin_path": list(i.glob('*' + channel + '.bin'))[0]}
                           for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

            test_set = SignalDataset(step=5000, window_size=window_size, bin_setup=test_config,cutoff=sample_rate//15,  source_dtype="float32")

            nn_config = load_yaml(Path("configs/nn_configs/"+model+".yaml"))
            neuro_net = NeuroNet(config=nn_config, metrics=True)
            neuro_net._model.load_state_dict(torch.load("trained_models/"+ channel+ "/"+model+".pt"))
            neuro_net._model.eval()
            neuro_net._model.to(DEVICE)
            test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)
            neuro_net._model.eval()
            with torch.no_grad():
                outputs = torch.empty(size=(0, 1), dtype=torch.float32, device="cuda").flatten()
                targets = torch.empty(size=(0, 1), dtype=torch.long, device="cuda").flatten()
                for i, (input, target) in enumerate(test_dataloader):
                    output = torch.from_numpy(neuro_net.predict(input.numpy(), argmax=True)).to("cuda")
                    target = target.to("cuda")
                    outputs = torch.cat((outputs, output), dim=0)
                    targets = torch.cat((targets, target), dim=0)
            neuro_net.plot_confusion_matrix(outputs.cpu().numpy(), targets.cpu().numpy())

            # print(classification_report(targets.cpu().numpy(), outputs.cpu().numpy()))
            print(f"Acuracy score for "+model+", channel "
                  + channel + f" is {accuracy_score(targets.cpu().numpy(), outputs.cpu().numpy())}")

DEVICE = "cuda"
window_size = 30000
sample_rate = 1562500
channel_1 = 'ch1'
channel_2 = 'ch2'
channel_3 = 'ch3'
channel_train = channel_3
channel_test = channel_3
models = ["CNN", "InceptionTime", "LSTM"]
channel_list = [channel_1, channel_2, channel_3]
evaluate_models(models, channel_list)