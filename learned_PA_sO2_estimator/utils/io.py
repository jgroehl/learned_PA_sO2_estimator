import os
import numpy as np
import tensorflow as tf
from learned_PA_sO2_estimator.utils import get_dataset_name, get_dataset_path
from learned_PA_sO2_estimator.utils.file_downloader import download_file


def load_data(dataset_id):
    dataset_path = get_dataset_path(dataset_id)
    if not os.path.exists(dataset_path):
        download_file(get_dataset_name(dataset_id))
    data = np.load(dataset_path)
    spectra = data["spectra"]
    spectra = (spectra - np.nanmean(spectra, axis=0)[np.newaxis, :]) / np.nanstd(spectra, axis=0)[
                                                                             np.newaxis, :]
    spectra = np.swapaxes(spectra, 0, 1)
    spectra = spectra.reshape((len(spectra), len(spectra[0]), 1))

    spectra = tf.convert_to_tensor(spectra)
    return spectra
