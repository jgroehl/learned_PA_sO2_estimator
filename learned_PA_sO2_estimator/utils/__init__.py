import os
import inspect
import numpy as np

WAVELENGTHS = [5, 10, 41]

DATASETS = [
    "ACOUS",
    "BASE",
    "BG_0-100",
    "BG-60-80",
    "BG_H2O",
    "HET_0-100",
    "HET-60-80",
    "ILLUM_5mm",
    "ILLUM_POINT",
    "INVIS_ACOUS",
    "INVIS",
    "INVIS_SKIN",
    "INVIS_SKIN_ACOUS",
    "MSOT_ACOUS",
    "MSOT",
    "MSOT_SKIN",
    "MSOT_SKIN_ACOUS",
    "RES_0.6",
    "RES_0.15",
    "RES_0.15_SMALL",
    "RES_1.2",
    "SKIN",
    "SMALL",
    "WATER_2cm",
    "WATER_4cm"
]


def get_dataset_name(dataset_identifier):
    return dataset_identifier + "_train.npz"


def get_dataset_path(dataset_identifier):
    base_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return f"{base_script_path}/../../data/{get_dataset_name(dataset_identifier)}"


def get_model_weights_name(num_wl, dataset_identifier):
    closest_wl = WAVELENGTHS[np.argmin(np.absolute(np.asarray(WAVELENGTHS) - num_wl))]
    return f"{dataset_identifier}_LSTM_{closest_wl}.h5"


def get_model_weights_path(num_wl, dataset_identifier):
    closest_wl = WAVELENGTHS[np.argmin(np.absolute(np.asarray(WAVELENGTHS)-num_wl))]
    base_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return f"{base_script_path}/../../data/{get_model_weights_name(closest_wl, dataset_identifier)}"
