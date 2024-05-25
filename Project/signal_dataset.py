import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
# from torch.utils.data import random_split
import torch
# import scipy.io.wavfile as wavfile
import re

# import random
# from itertools import chain

DEVICE = 'cuda'


# ctrl alt O


# Function to get key and value by index

class SignalDataset(Dataset):

    def __init__(self, step: int, window_size: int, bin_setup: list,
                 device="cpu", source_dtype="float32"):

        self.step = step
        self.window_size = window_size
        self.device = device
        self.source_dtype = source_dtype
        self.bin_setup = bin_setup

        self.loaded_signal = {}

        self.sampling = []
        self.num_of_samples = []

        self.indices = [
            # ('class', sig_id, sample_id)
        ]

        self.label_set = set()
        self.label_dict = {}

        self._load_signals()  # loads raw data in form of numpy arrray in to a list
        self._create_index_setup()  # creates indices of starts of intervals

    def _load_signals(self):
        for bin_config in self.bin_setup:
            label = bin_config['label']
            i_min, i_max = bin_config['interval']
            if label not in self.loaded_signal:
                self.loaded_signal[label] = []

            self.loaded_signal[label].append(np.fromfile(bin_config['bin_path'],
                                                         dtype=self.source_dtype)[i_min: i_max])  # interval

    def _create_index_setup(self):
        for label, signal_list in self.loaded_signal.items():
            self.label_set.add(label)
            for sig_id, s in enumerate(signal_list):
                for sample_id in range(0, len(s) - self.window_size, self.step):
                    self.indices.append((label, sig_id, sample_id))

        for i, label in enumerate(sorted(self.label_set)):
            self.label_dict[label] = i

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label, sig_id, sample_id = self.indices[idx]

        return (self.loaded_signal[label][sig_id][sample_id: sample_id + self.window_size].reshape(1 , -1),
                self.label_dict[label])

