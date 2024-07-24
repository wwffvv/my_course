import numpy as np
import os

for file in os.listdir("."):
    if file.endswith(".npz"):
        metric = np.load(f"{file}", allow_pickle=True)
        metric_score = metric['score']
        print(f"{file}: {metric_score}")