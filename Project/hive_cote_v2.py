import re
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset.signal_dataset import SignalDataset
import seaborn as sns



def average_pooling(data, pool_size):
    pooled_data = []
    for i in range(0, len(data), pool_size):
        pooled_data.append(np.mean(data[i:i+pool_size]))
    return np.array(pooled_data)

def plot_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, save_path: Path = None):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    df_cm = pd.DataFrame(cm, index=[i for i in range(len(classes))],
                         columns=[i for i in range(len(classes))])
    df_cm_norm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in range(len(classes))],
                              columns=[i for i in range(len(classes))])
    plt.figure(figsize=(3.3, 3))
    sns.heatmap(df_cm_norm, annot=df_cm, cbar=False, xticklabels=classes, yticklabels=classes, fmt=".0f")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


window_size = 30000
sample_rate = 1562500
channel_1 = 'ch1'
channel_2 = 'ch2'
channel_3 = 'ch3'
channel_train = channel_3
channel_test = channel_3

signal_data_dir = "/mnt/home2/Motor_project/AE_PETR_loziska/"

# TRAIN
train_config = [{"label": (int(i.stem) - 1) // 4,
                 "channels": len(list(i.glob('*' + channel_train + '.bin'))),
                 "interval": [0, int(4 * sample_rate)],
                 "bin_path": list(i.glob('*' + channel_train + '.bin'))[0]}
                for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

# validation_config = [{"label": (int(i.stem) - 1) // 4,
#                  "channels": len(list(i.glob('*' + channel_train + '.bin'))),
#                  "interval": [int(4 * sample_rate), int(4.5 * sample_rate)],
#                  "bin_path": list(i.glob('*' + channel_train + '.bin'))[0]}
#                 for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]

test_config = [{"label": (int(i.stem) - 1) // 4,
                "channels": len(list(i.glob('*' + channel_test + '.bin'))),
                "interval": [int(4.5 * sample_rate), int(5 * sample_rate)],
                "bin_path": list(i.glob('*' + channel_test + '.bin'))[0]}
               for i in Path(signal_data_dir).glob('*') if re.search(r'\d$', i.stem)]


cutoff = (1000,sample_rate // 60)


train_set = SignalDataset(step=5000, window_size=window_size, cutoff=cutoff, bin_setup=train_config, source_dtype="float32")
test_set = SignalDataset(step=5000, window_size=window_size, bin_setup=test_config,cutoff=cutoff,   source_dtype="float32")



pool = 30
X_train = np.asarray([average_pooling(data[0].reshape(-1),pool) for data in train_set])
y_train = np.asarray([data[1] for data in train_set])
X_test = np.asarray([average_pooling(data[0].reshape(-1),pool) for data in test_set])
y_test = np.asarray([data[1] for data in test_set])

classes = ["IL", "CD", "EL", "CL", "OS", "NEW"]




save_path = "/home/petr/Documents/bachelor_project/Project/trained_models/OTHER/"

hive = HIVECOTEV2(verbose=1)



print("starting HIVE:", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))


hive.fit(X_train, y_train)

y_pred = hive.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
plot_confusion_matrix(y_pred, y_test, save_path=save_path+"CMhive_ch3.pdf")
print(f"HC2: Accuracy: {accuracy:.4f}")












