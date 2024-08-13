# Bearing Damage Classification using Neural Networks

This project is part of my bachelor's thesis. The main focus of this thesis is the classification of acoustic emission (AE) signals of damaged bearings. Six selected damage classes were considered, and the signals were recorded using three different sensors, resulting in three distinct datasets, **D₁**, **D₂**, and **D₃**. The study aimed to answer several key questions regarding sensor suitability, the impact of surrounding noise, and the generalization capability of the model across different sensors. This was the first major project I've ever worked on, and any corrections to my code are more than welcome. The link to the thesis will be provided once it is published. 

## Models Used

- **CNN**: Convolutional Neural Network classifier.
- **InceptionTime**: SOTA neural network time series classifier.
- **LSTM**: Long Short-Term Memory network classifier.
- **HIVE-COTE v2 (HC2)**: SOTA time-series classifier used as a benchmark.


## Project Structure

- **configs/**: Configuration files for neural networks and other models.
- **dataset/**: Scripts for dataset handling and augmentation.
- **hyperparameter_optimization/**: Scripts for optimizing model hyperparameters.
- **networks/**: Implementations of different neural network architectures.
- **signal_model.py**: Class handling all network loading, training, evaluation, visualization, etc.
- **raw_ae.py**: Script to run the first experiment - the classification of recorded acoustic emission of bearings.
- **gearbox_emmision.py**: Script to run for the second experiment involving the addition of acoustic emission of the gearbox.
- **cross_channel.py**: Script to run the third experiment that addresses a possible simple approach to cross-channel generalization.
- **hive_cote_v2.py**: Script to run the HC2 classifier.


### Conda env

Once you have Conda installed and have navigated to the project directory, create a new environment using the `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

This command will create a new Conda environment with all the dependencies specified in the `environment.yaml` file.


## Results Summary

We recommend you skim through the paper to understand the presented results better. In paper are also provided the training and validation accuracy curves and confusion matrices. 

### Experiment 1: Classification of AE Signals from Damaged Bearings

This experiment focused on classifying raw AE signals from damaged bearings.

| Model | **D₁**(%) | **D₂**(%) | **D₃**(%) | Training time (min) | Inference time (s) |
|-------|-----------|-----------|-----------|---------------------|--------------------|
| CNN   | 98.73     | 98.62     | 93.03     | 7.5                 | 5.0                |
| LSTM  | 97.55     | 97.21     | 90.90     | 13.3                | 4.2                |
| IT    | **99.63** | **99.45** | **98.39** | 81.2                | 3.5                |
| HC2   | 95.59     | 95.95     | 86.52     | 756.1               | 43.4               |

**Table 1**: Test accuracies of classification of damaged bearings. The training and inference time values were averaged across all three datasets. Highlighted are the best results for each dataset.

### Key Observations

- **Accuracy**: All proposed deep learning models outperformed HC2, particularly on **D₃**.
- **Sensors**: Datasets **D₁** and **D₂** produced similar results, suggesting that both Sensor 1 and Sensor 2 are suitable for practical applications.

### Experiment 1: Frequency Domain Classification

As an extension of Experiment 1, signals were transformed into the frequency domain using the Fast Fourier Transform (FFT). The results are shown below:

| Model | **D₁**(%) | **D₂**(%) | **D₃**(%) |
|-------|-----------|-----------|-----------|
| CNN   | **99.02** | 96.38     | 95.86     |
| LSTM  | 86.61     | 85.81     | 83.61     |
| IT    | 97.98     | **98.78** | **98.51** |

**Table 2**: Test accuracies of models in the frequency domain.

- **Findings**: Transforming signals into the frequency domain generally led to lower accuracy, except for minor improvements in specific cases.

### Experiment 2: Addition of Gearbox AE

This experiment simulated real-world conditions by adding gearbox AE to the bearing signals. The models were trained on signals with added gearbox AE to evaluate their performance. Training directly on noisy data was necessary for achieving any level of generalization.

| Model | **D₁** with Gearbox (%) |
|-------|-------------------------|
| CNN   | 75.38                   |
| LSTM  | **75.72**               |
| IT    | 73.76                   |

**Table 3**: Test accuracies of models trained on signals with added gearbox emission.

- **Results**: The addition of gearbox noise reduced accuracy across all models, as expected. There are concerns about noise from heavier machinery, which is much more noisy. 

### Experiment 3: Cross-Channel Generalization

The final experiment tested the models' ability to generalize across different sensors, using min-max scaling and data augmentation to improve performance.

| Model         | **D₁**(%) | **D₂**(%) | **D₃**(%) | **D₁**(%) | **D₂**(%) | **D₃**(%) | **D₁**(%) | **D₂**(%) | **D₃**(%) |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **Trained on** | **D₁** | **D₁** | **D₁** | **D₂** | **D₂** | **D₂** | **D₃** | **D₃** | **D₃** |
| CNN           | 94.62     | 50.30     | 22.72     | 35.73     | 93.29     | 37.52     | 45.21     | 38.93     | 88.05     |
| LSTM          | 90.01     | 46.42     | 31.73     | 32.31     | 86.61     | 34.09     | 35.13     | 41.03     | 82.32     |
| IT            | 95.36 | 41.17     | 56.12 | 42.15     | 93.06| 30.00     | **50.76** | **57.44** | 83.59 |

**Table 4**: Cross-channel generalization accuracies across different datasets.

- **Findings**: The best results were obtained with the IT model trained on **D₃**. Generalization across channels remains challenging and a more sophisticated approach would be appropriate.

