from deepface import DeepFace
import os
from os import path
from itertools import combinations
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json

## Loops over subfolders inside main_dir, with each subfolder containing images of an individual
## implement logging and parallelism 

main_dir = "images"

# file names for individuals ['0', '1'] etc
individual_dir = [f.name for f in os.scandir(main_dir) if f.is_dir()]

results = {}

# loop over each individual in each folder
for idx in individual_dir:
    folder_path = os.path.join(main_dir, idx)

    # get image files
    image_files = sorted(os.listdir(folder_path))

    # get combinatations
    image_pairs = list(combinations(image_files, 2))

    TP = 0
    FN = 0

    for img1, img2 in image_pairs:
        result = DeepFace.verify(
            img1_path = os.path.join(folder_path, img1),
            img2_path = os.path.join(folder_path, img2),
            enforce_detection=False
        )

        is_match = result['verified']

        if is_match:
            TP += 1
        else:
            FN += 1
    
    TPR = TP / (TP + FN)
    FNR = 1 - TPR
    results[idx] = FNR

avg_FNR = np.mean(list(results.values()))
results["mean_FNR"] = avg_FNR

for idx, fnr in results.items():
    print(f'FNR for individual {idx} is: {fnr:.2%}')

print(f'Mean FNR across all individuals in group {main_dir}: {avg_FNR:.2%}')
