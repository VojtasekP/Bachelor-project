from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

cuda0 = torch.device('cuda:0')

# ctrl + alt + o

class CustomDict:
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if key in self._data:
            if isinstance(value, np.ndarray):
                self._data[key] = np.concatenate((self._data[key], value))
            else:
                raise TypeError("Only numpy arrays can be added to existing numpy arrays.")
        else:
            if isinstance(value, np.ndarray):
                self._data[key] = value
            else:
                raise TypeError("Values must be numpy arrays.")

    def __delitem__(self, key):
        del self._data[key]

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self):
        return repr(self._data)


# Function to get key and value by index

def get_position_label_start_by_index(index, list_of_dicts: list[CustomDict]):
    # Initialize cumulative length
    cumulative_length = 0

    # Iterate through dictionaries in the list
    for i, dictionary in enumerate(list_of_dicts):
        # Iterate through key-value pairs in each dictionary
        for key, value in dictionary.items():
            # Calculate the cumulative length
            length = len(value)
            if index < cumulative_length + length:
                return i, key, value[index - cumulative_length]
            cumulative_length += length

    # If index is out of range, return None
    # return None, None, None


class SignalDataset(Dataset):

    def __init__(self, step: int, window_size: int, paths_and_classes: dict, device=torch.device("cpu")):
        source_dtype = "float32"
        self.step = step
        self.window_size = window_size
        self.device = device

        paths = list(paths_and_classes)  # creates list of paths
        class_distribution = list(paths_and_classes.values())

        self._load_signals(paths, source_dtype)  # loads raw data in form of numpy array in to a list
        self._create_sampling(class_distribution)  # creates indices of starts of intervals

    def _load_signals(self, paths: list, dtype="float32"):
        self.loaded_signal = []
        for i, path in enumerate(paths):
            # count = np.fromfile(path, dtype=dtype).shape[0]
            self.loaded_signal.append(np.fromfile(path, dtype=dtype))

    def _create_sampling(self, class_distributions: list):

        self.sampling = []
        self.num_of_samples = []

        for i, distribution in enumerate(class_distributions):

            assert len(distribution["size"]) == len(distribution["labels"]), "labels and sizes are mismatched"

            full_length = np.sum(distribution["size"])
            recording_size = self.loaded_signal[i].shape[0]  # number of points
            already_binned = 0
            shift = 4 * self.window_size  # later will be dependent on frequency of sensor
            bins_per_label = CustomDict()

            for idx in range(len(distribution["labels"])):
                # later will be size calculated based on the frequency and derived from actual seconds recorded
                fraction_of_sample = distribution["size"][idx] / full_length
                length_of_sample = int(np.floor(fraction_of_sample * recording_size))

                start = already_binned + shift
                end = length_of_sample + already_binned - shift
                bins = np.arange(start=start, stop=end, step=self.step)
                bins_per_label[distribution["labels"][idx]] = bins
                self.num_of_samples.append(bins.size)

                already_binned += length_of_sample
            self.sampling.append(bins_per_label)

    # 0:      26100000, 26171875: 52271875
    # 400000: 25700000, 26571875: 51871875

    def __len__(self):
        return np.sum(self.num_of_samples)

    def __getitem__(self, idx):
        position, label, start = get_position_label_start_by_index(idx, self.sampling)
        return torch.tensor(self.loaded_signal[position][start: start + self.window_size]), torch.tensor(label)


directory_of_signal_data = "/home/petr/Documents/Motor_projekt/Data"
list_of_paths = sorted(Path(directory_of_signal_data).glob('*.bin'))
list_of_classes = [{"labels": [0, 10], "size": [15, 15]},
                   {"labels": [1, 11], "size": [15, 15]},
                   {"labels": [2, 12], "size": [15, 15]},
                   {"labels": [3, 13], "size": [15, 15]},
                   {"labels": [4, 14], "size": [15, 15]},
                   {"labels": [5, 15], "size": [15, 15]},
                   {"labels": [6, 16], "size": [15, 15]},
                   {"labels": [7, 17], "size": [15, 15]},
                   {"labels": [8, 18], "size": [15, 15]}]

data_specs = dict(zip(list_of_paths, list_of_classes))
"""

"""

dataset = SignalDataset(step=1000, window_size=1000, paths_and_classes=data_specs, device=cuda0)


print(dataset[len(dataset)-1])
