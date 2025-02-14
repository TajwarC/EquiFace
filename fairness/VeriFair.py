import os
import numpy as np
from itertools import combinations
from tqdm import tqdm
from deepface import DeepFace

def FNR(dataset_dir) -> str:
    # Initiate results
    results = {}

    # Main directory
    main_dir = dataset_dir

    # Individual directories
    individual_dirs = [f.name for f in os.scandir(main_dir) if f.is_dir()]
    
    # Precompute total number of comparisons
    total_pairs = 0
    image_pairs_dict = {}

    for idx in individual_dirs:
        folder_path = os.path.join(main_dir, idx)
        image_files = sorted(os.listdir(folder_path))
        image_pairs = list(combinations(image_files, 2))
        image_pairs_dict[idx] = (folder_path, image_pairs)
        total_pairs += len(image_pairs)

    print(f'{total_pairs} total input pairs from {len(individual_dirs)} IDs.')

    FN = 0
    TP = 0

    # Single progress bar for all comparisons
    with tqdm(total=total_pairs, desc="Verifying input pairs for each ID", unit="pair") as pbar:
        for idx, (folder_path, image_pairs) in image_pairs_dict.items():
            for img1, img2 in image_pairs:
                result = DeepFace.verify(
                    img1_path=os.path.join(folder_path, img1),
                    img2_path=os.path.join(folder_path, img2),
                    enforce_detection=False
                )

                is_match = result['verified']
                if is_match:
                    TP += 1
                else:
                    FN += 1

                pbar.update(1)  # Update progress bar after each comparison

            # Calculate FNR for each individual
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            results[idx] = 1 - TPR

    # Calculate mean FNR across all individuals
    avg_FNR = np.mean(list(results.values())) if results else 0
    results["mean_FNR"] = avg_FNR

    print(f'Mean FNR across all IDs in group {main_dir}: {avg_FNR:.2%}')
    
    return results

    

def FPR(dataset_dir) -> str:
    subfolders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])

    # Precompute total comparisons
    total_comparisons = 0
    image_counts = {}

    # Count images in each subfolder
    for folder in subfolders:
        path = os.path.join(dataset_dir, folder)
        image_counts[folder] = len([img for img in os.listdir(path) if img.endswith('.png')])

    # Compute total number of comparisons before running verification
    for folder1, folder2 in itertools.combinations(subfolders, 2):
        total_comparisons += image_counts[folder1] * image_counts[folder2]

    print(f"Total number of input pairs: {total_comparisons}")

    # Initiate metrics
    FP = 0
    TN = 0

    # Progress bar 
    progress_bar = tqdm(total=total_comparisons, desc="Processing input pairs", unit="pair")

    # Run inference on input pairs
    for folder1, folder2 in itertools.combinations(subfolders, 2):
        path1 = os.path.join(dataset_dir, folder1)
        path2 = os.path.join(dataset_dir, folder2)

        images1 = [os.path.join(path1, img) for img in os.listdir(path1) if img.endswith('.png')]
        images2 = [os.path.join(path2, img) for img in os.listdir(path2) if img.endswith('.png')]

        for img1 in images1:
            for img2 in images2:
                result = DeepFace.verify(img1, img2, enforce_detection=False)

                if result["verified"]:
                    FP += 1
                else:
                    TN += 1
                
                progress_bar.update(1) 

    progress_bar.close()

    # Calculate metrics
    FPR = FP / total_comparisons if total_comparisons > 0 else 0
    TNR = TN / total_comparisons if total_comparisons > 0 else 0

    # Print results
    print(f"False Positives: {FP}")
    print(f"True Negatives: {TN}")
    print(f"False Positive Rate (FPR): {FPR:.2f}")
    print(f"True Negative Rate (TNR): {TNR:.2f}")


