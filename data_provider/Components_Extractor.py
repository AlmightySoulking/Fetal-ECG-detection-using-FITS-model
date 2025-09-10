from .old_Fetal_data_loader import data_loader
import scipy.signal as sig
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt # Import specific functions from scipy.signal
import pandas as pd
from pywt import wavedec, waverec, threshold
import numpy as np
from sklearn.decomposition import FastICA

def bandpass_filter(self, data, lowcut=1.0, highcut=50.0, fs=1000.0, order=4):
    """
    Apply a bandpass filter to the ECG signal to remove noise
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def notch_filter(self, data, notch_freq=50.0, quality_factor=30.0, fs=1000.0):
    """
    Apply a notch filter to remove power line interference
    """
    # Use scipy.signal explicitly
    b, a = sig.iirnotch(notch_freq, quality_factor, fs)
    y = sig.filtfilt(b, a, data)
    return y


def preprocess_ecg(self, data, fs=1000.0):
    # Convert to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            signal_data = data.values.flatten()
        else:
            signal_data = data.iloc[:, -1].values
    else:
        signal_data = data

    # Normalize
    signal_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)

    # First apply a wider bandpass to preserve overall signal
    signal_data = self.bandpass_filter(signal_data, lowcut=0.5, highcut=150.0, fs=fs)

    # Remove power line interference
    signal_data = self.notch_filter(signal_data, notch_freq=50.0, fs=fs)
    signal_data = self.notch_filter(signal_data, notch_freq=60.0, fs=fs)

    # Advanced baseline correction using polynomial fitting
    t = np.arange(len(signal_data))
    baseline = sig.savgol_filter(signal_data, window_length=int(fs * 1.5) | 1, polyorder=2)
    signal_data = signal_data - baseline

    # Apply wavelet denoising for better preservation of QRS complexes

    coeffs = wavedec(signal_data, 'db4', level=5)
    for i in range(1, len(coeffs)):
        coeffs[i] = threshold(coeffs[i], np.std(coeffs[i]) * 0.1, mode='soft')
    signal_data = waverec(coeffs, 'db4')

    return signal_data


def extract_maternal_fetal_components(self, ecg_data, fs=1000.0):
    """Improved component separation"""
    if len(ecg_data.shape) == 1:
        # Create a matrix with delayed versions
        delay = 20  # Increase delay for better separation
        rows = len(ecg_data) - delay
        X = np.zeros((rows, delay))
        for i in range(delay):
            X[:, i] = ecg_data[i:i + rows]
    else:
        X = ecg_data

    # Apply ICA with more components
    ica = FastICA(n_components=min(X.shape[1], 5), max_iter=1000)
    S = ica.fit_transform(X)

    # Identify maternal vs. fetal components based on frequency content
    maternal_idx = None
    fetal_idx = []

    for i in range(S.shape[1]):
        # Calculate power spectrum
        f, Pxx = sig.welch(S[:, i], fs=fs, nperseg=1024)

        # Get dominant frequency
        dominant_freq_idx = np.argmax(Pxx[1:]) + 1  # Skip DC component
        dominant_freq = f[dominant_freq_idx]

        # Maternal heart rate typically 60-100 bpm = 1-1.7 Hz
        # Fetal heart rate typically 110-160 bpm = 1.8-2.7 Hz
        if 1.0 <= dominant_freq <= 1.7:
            maternal_idx = i
        elif 1.8 <= dominant_freq <= 3.0:
            fetal_idx.append(i)

    # If you couldn't identify by frequency, fall back to traditional method
    if maternal_idx is None:
        maternal_idx = 0

    if not fetal_idx:
        fetal_idx = [i for i in range(S.shape[1]) if i != maternal_idx][:2]  # Use top 2 non-maternal components

    maternal_component = S[:, maternal_idx]
    fetal_signal = np.sum(S[:, fetal_idx], axis=1) if len(fetal_idx) > 0 else S[:, 1] if S.shape[1] > 1 else np.zeros_like(S[:, 0])

    return maternal_component, fetal_signal

if __name__ == '__main__':
    pass