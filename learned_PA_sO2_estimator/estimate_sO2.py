import os
import numpy as np
from learned_PA_sO2_estimator.models import LSTMParams
from learned_PA_sO2_estimator.utils import get_model_weights_path, get_model_weights_name
from learned_PA_sO2_estimator.utils.file_downloader import download_file


def evaluate(dataset, train_dataset_id, batch_size=50000):
    """

    :param dataset:
        This function assumes that _dataset is a numpy array with shape
        (N, M), where N is the number of spectra and M is the number of
        wavelengths.
    :param train_dataset_id:
        The identifier of the training dataset that the model should be trained on for this analysis.
    :param batch_size:
        int describing the number of spectra fed through the network at once. If you have a particularly small amount
        of RAM or VRAM it might be necessary to reduce this number.
    :return:
        a numpy array in the same shape as the input array.
    """

    num_samples, num_wavelengths, num_batches = np.shape(dataset)

    model_weights_path = get_model_weights_path(num_wavelengths, train_dataset_id)

    if not os.path.exists(model_weights_path):
        download_file(get_model_weights_name(num_wavelengths, train_dataset_id))

    _model = LSTMParams.load(model_weights_path)
    _model.compile()

    results = []
    for i in range(num_samples // batch_size):
        result = _model(dataset[i * batch_size:(i + 1) * batch_size])
        results.append(result.numpy())
    i = num_samples // batch_size
    results.append(_model(dataset[i * batch_size:]).numpy())
    result = np.vstack(results)

    return result
