
import numpy as np
from torch.utils.data import Dataset
from dataset.data_augmentation import augment_data
from dataset.data_augmentation import bandpass_filter, detrend_signal

class SignalDataset(Dataset):

    def __init__(self, step: int, window_size: int, bin_setup: list, aug_params: dict = {}, cutoff:tuple= None,
                 noise: dict = None, seed= None, device="cpu", source_dtype="float32"):

        self.step = step
        self.window_size = window_size
        self.device = device
        self.source_dtype = source_dtype
        self.bin_setup = bin_setup
        self.aug_params = aug_params
        self.cutoff = cutoff
        self.seed = seed
        self.loaded_signal = {}
        self.noise = noise
        self.sampling = []
        self.num_of_samples = []

        self.indices = [
            # ('class', sig_id, sample_id)
        ]

        self.label_set = set()
        self.label_dict = {}

        self._load_signals()  # loads raw data in form of numpy arrray in to a list

        if self.noise is not None:

            self._load_noise()

        self._create_index_setup()  # creates indices of starts of intervals


    def _load_signals(self):
        for bin_config in self.bin_setup:
            label = bin_config['label']

            if label not in self.loaded_signal:
                self.loaded_signal[label] = []
            if bin_config["interval"] is not None:
                i_min, i_max = bin_config['interval']
                signal =np.fromfile(bin_config['bin_path'], dtype=self.source_dtype)[i_min:i_max]
                signal = detrend_signal(signal)
                # signal = augment_data(signal, augmentation_params=self.aug_params)
                self.loaded_signal[label].append(bandpass_filter(signal, cutoff=self.cutoff, sample_rate=1562500))
            else:
                self.loaded_signal[label].append(np.fromfile(bin_config['bin_path'],
                                                             dtype=self.source_dtype))
    def _load_noise(self):
        self.loaded_noise = self.noise['intensity']*bandpass_filter(np.fromfile(self.noise['bin_path'], dtype=self.source_dtype), cutoff=self.cutoff, sample_rate=1562500)

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
        start_index = sig_id * 500000

        if self.seed is not None:
            np.random.seed(self.seed + idx)

        if self.noise is not None:
            return ((self.loaded_signal[label][sig_id][sample_id: sample_id + self.window_size] +
                    self.loaded_noise[start_index + sample_id: start_index + sample_id + self.window_size]).reshape(1 , -1),
                    self.label_dict[label])

        return (self.loaded_signal[label][sig_id][sample_id: sample_id + self.window_size].reshape(1 , -1),
                self.label_dict[label])


