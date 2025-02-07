from deepface import DeepFace
import os
from os import path
import itertools
from itertools import combinations
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json

## calculates false positive rate over entire demographic group in image dataset
## implement logging and parallelism 

main_dir = "images"
subfolders = sorted([f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))])

# initiate metrics 

false_positives = 0
true_negatives = 0
total_comparisons = 0

# find every combination between all images in the main_dir
for folder1, folder2 in itertools.combinations(subfolders, 2):
    path1 = os.path.join(main_dir, folder1)
    path2 = os.path.join(main_dir, folder2)

    # list the combinations
    images1 = [os.path.join(path1, img) for img in os.listdir(path1) if img.endswith(('.png'))]
    images2 = [os.path.join(path2, img) for img in os.listdir(path2) if img.endswith(('.png'))]

    # Compare each image in folder1 with each image in folder2
    for img1 in images1:
        for img2 in images2:
            result = DeepFace.verify(img1, img2, enforce_detection=False)
            total_comparisons += 1

            if result["verified"]:
                false_positives += 1
            else: 
                true_negatives += 1

# calculate metrics 
fpr = false_positives / total_comparisons
tnr = true_negatives / total_comparisons

print(f"Total Comparisons: {total_comparisons}")
print(f"False Positives: {false_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Positive Rate (FPR): {fpr:.2f}")
print(f"True Negative Rate (TNR): {tnr:.2f}")
