import os
import numpy as np
import tensorflow as tf
from learned_PA_sO2_estimator.utils import get_dataset_name, get_dataset_path
from learned_PA_sO2_estimator.utils.file_downloader import download_file


def load_data(dataset_id, data_wavelengths):
    dataset_path = get_dataset_path(dataset_id)
    if not os.path.exists(dataset_path):
        download_file(get_dataset_name(dataset_id))
    data = np.load(dataset_path)
    spectra = data["spectra"]
    spectra = zero_pad_wavelengths(spectra, data_wavelengths)
    spectra = np.swapaxes(spectra, 0, 1)
    spectra = spectra.reshape((len(spectra), len(spectra[0]), 1))

    spectra = tf.convert_to_tensor(spectra)
    return spectra


def zero_pad_wavelengths(spectra, data_wavelengths):
    """
    :param data:
        data is a numpy array of shape (num_samples, num_wavelengths)
    :param wavelengths:
        a list of length num_wavelengths containing wavelengths
    :return:
    """

    wavelengths = list(np.arange(700, 901, 5))
    wl_mask = [wl in wavelengths for wl in data_wavelengths]
    inv_wl_mask = np.invert(wl_mask)
    spectra[inv_wl_mask, :] = np.nan
    spectra = (spectra - np.nanmean(spectra, axis=0)[np.newaxis, :]) / np.nanstd(spectra, axis=0)[np.newaxis, :]
    full_spectra = np.zeros((41, len(spectra[0])))
    full_spectra[wl_mask, :] = spectra[wl_mask, :]
    spectra = full_spectra

    return spectra

def convert_numpy_array(data, data_wavelengths):
    """
    :param data:
        data is a numpy array of shape (num_samples, num_wavelengths)
    :param wavelengths:
        a list of length num_wavelengths containing wavelengths
    :return:
    """

    data = zero_pad_wavelengths(data, data_wavelengths)
    data = np.swapaxes(data, 0, 1)
    data = data.reshape((len(data), len(data[0]), 1))
    data = tf.convert_to_tensor(data)
    return data

