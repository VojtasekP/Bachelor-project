import numpy as np
from scipy.signal import resample



DEVICE = 'cuda'
def time_shift(signal, shift):
    return np.roll(signal, shift)

def add_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def scale_signal(signal, scale_factor):
    return signal * scale_factor

def time_stretch(signal, stretch_factor):
    length = len(signal)
    stretched = resample(signal, int(length * stretch_factor))
    if len(stretched) > length:
        return stretched[:length]
    else:
        return np.pad(stretched, (0, length - len(stretched)), 'constant')
# Function to get key and value by index
def lowpass_filter(signal, cutoff_freq, sample_rate):
    if cutoff_freq is None:
        return signal
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    lowpass_filter = np.abs(freqs) <= cutoff_freq
    filtered_fft_signal = fft_signal * lowpass_filter
    filtered_signal = np.fft.ifft(filtered_fft_signal)

    return filtered_signal.real.astype(np.float32)

def frequency_shift(signal, frequency_shift_hz):

    fs = 1562500
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    shift_phase = np.exp(1j * 2 * np.pi * frequency_shift_hz * freqs)
    shifted_fft_signal = fft_signal * shift_phase
    shifted_signal = np.fft.ifft(shifted_fft_signal)
    shifted_signal = np.real(shifted_signal)

    return shifted_signal.astype(np.float32)

def magnitude_warping(signal, warp_factor):

    mag_spectrum = np.abs(np.fft.fft(signal))
    warped_mag_spectrum = mag_spectrum ** warp_factor
    phase_spectrum = np.angle(np.fft.fft(signal))
    warped_spectrum = warped_mag_spectrum * np.exp(1j * phase_spectrum)
    warped_signal = np.fft.ifft(warped_spectrum)
    warped_signal = np.real(warped_signal)

    return warped_signal.astype(np.float32)

def augment_data(data, augmentation_params = None):
    if augmentation_params is None:
        return data
    if 'time_shift' in augmentation_params:
        shift = np.random.randint(-augmentation_params['time_shift'], augmentation_params['time_shift'])
        data = time_shift(data, shift)
    if 'add_noise' in augmentation_params:
        data = add_noise(data, augmentation_params['add_noise'])
    if 'scale' in augmentation_params:
        scale_factor = np.random.uniform(1-augmentation_params['scale'], 1+augmentation_params['scale'])
        data = scale_signal(data, scale_factor)
    if 'time_stretch' in augmentation_params:
        stretch_factor = np.random.uniform(1-augmentation_params['time_stretch'], 1+augmentation_params['time_stretch'])
        data = time_stretch(data, stretch_factor)
    if "frequency_shift" in augmentation_params:
        frequency_shift_hz = np.random.uniform(-augmentation_params["frequency_shift"], augmentation_params["frequency_shift"])
        data = frequency_shift(data, frequency_shift_hz)
    if "magnitude_shift" in augmentation_params:
        magnitude_shift = augmentation_params['magnitude_shift']
        data = magnitude_warping(data, magnitude_shift)
    return data.astype(np.float32)
# ctrl alt O
