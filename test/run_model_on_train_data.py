import numpy as np
from learned_PA_sO2_estimator.utils import get_dataset_path
from learned_PA_sO2_estimator.utils.io import load_data
from learned_PA_sO2_estimator.estimate_sO2 import evaluate
import matplotlib.pyplot as plt


data = load_data("BASE", list(np.arange(700, 901, 5)))
gt = np.load(get_dataset_path("BASE"))["oxygenation"]

sO2 = evaluate(data, "BASE")

plt.scatter(gt, sO2, alpha=0.002)
plt.show()





