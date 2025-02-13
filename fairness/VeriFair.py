from deepface import DeepFace
import os
from os import path
from itertools import combinations
import numpy as np

def FNR(dataset_dir) -> str:
     
    # initiate results

    results = {}

    # Main directory
    main_dir = dataset_dir

    # Individual file names
    individual_dir = [f.name for f in os.scandir(main_dir) if f.is_dir()]

    # for each identity find the images and get combinations 

    for idx in individual_dir:
        folder_path = os.path.join(main_dir, idx)
        image_files = sorted(os.listdir(folder_path))
        image_pairs = list(combinations(image_files, 2))
        num_pairs = len(image_pairs)

        # Initiate FPs and TPs

        FN = 0
        TP = 0
        
        # Verify all input pairs for each ID
        for img1, img2 in image_pairs:
            result = DeepFace.verify(
                img1_path = os.path.join(folder_path, img1),
                img2_path = os.path.join(folder_path, img2),
                enforce_detection = False
            )

            is_match = result['verified']

            if is_match:
                TP =+ 1
            else:
                FN =+ 1

        TPR = TP / (TP + FN)

        FNR = 1-TPR

        results[idx] = FNR
        avg_FNR = np.mean(list(results.values()))
        results["mean_FNR"] = avg_FNR
        
    print(f'Mean FNR across all individuals in group {main_dir}: {avg_FNR:.2%}')
    

def FPR(dataset_dir)->str:
    subfolders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])

    # initiate metrics 

    FP = 0
    TN = 0
    total_comparisons = 0

    # find every combination between all images in the main_dir
    for folder1, folder2 in itertools.combinations(subfolders, 2):
        path1 = os.path.join(dataset_dir, folder1)
        path2 = os.path.join(dataset_dir, folder2)

        # list the combinations
        images1 = [os.path.join(path1, img) for img in os.listdir(path1) if img.endswith(('.png'))]
        images2 = [os.path.join(path2, img) for img in os.listdir(path2) if img.endswith(('.png'))]

        # Compare each image in folder1 with each image in folder2
        for img1 in images1:
            for img2 in images2:
                result = DeepFace.verify(img1, img2, enforce_detection=False)
                total_comparisons += 1

                if result["verified"]:
                    FP += 1
                else: 
                    TN += 1

    # calculate metrics 
    FPR = FP / total_comparisons
    TNR = TN / total_comparisons

    # Print results
    print(f"Total Comparisons: {total_comparisons}")
    print(f"False Positives: {FP}")
    print(f"True Negatives: {TN}")
    print(f"False Positive Rate (FPR): {FPR:.2f}")
    print(f"True Negative Rate (TNR): {TNR:.2f}")