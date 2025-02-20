import os
import numpy as np
from itertools import combinations
from tqdm import tqdm
from deepface import DeepFace
from multiprocessing import Pool, cpu_count

def verify_pair(args):
    """Function to verify a pair of images and return result."""
    folder_path, img1, img2 = args
    result = DeepFace.verify(
        img1_path=os.path.join(folder_path, img1),
        img2_path=os.path.join(folder_path, img2),
        enforce_detection=False,
    )
    return result['verified']

def FNR(dataset_dir, use_multiprocessing=False, num_cores=None):
    
    if use_multiprocessing:
        if num_cores is None:
            raise ValueError("You must specify the number of cores you want to use when use_multiprocessing=True")
        if num_cores < 1 or num_cores > cpu_count():
            raise ValueError(f"num_cores must be between 1 and {cpu_count()}")
    # Initiate results
    results = {}

    # Main directory
    main_dir = dataset_dir

    # Individual directories
    individual_dirs = [f.name for f in os.scandir(main_dir) if f.is_dir()]
    
    # Precompute total number of comparisons
    total_pairs = 0
    image_pairs_list = []

    for idx in individual_dirs:
        folder_path = os.path.join(main_dir, idx)
        image_files = sorted(os.listdir(folder_path))
        image_pairs = list(combinations(image_files, 2))
        total_pairs += len(image_pairs)
        image_pairs_list.extend([(folder_path, img1, img2) for img1, img2 in image_pairs])

    print(f'{total_pairs} total input pairs from {len(individual_dirs)} IDs.')

    FN = 0
    TP = 0

    # Use multiprocessing to speed up the verification
    if use_multiprocessing:
        with Pool(num_cores) as pool, tqdm(total=total_pairs, desc="Processing input pairs", unit="pair") as pbar:
            for is_match in pool.imap_unordered(verify_pair, image_pairs_list):
                if is_match:
                    TP += 1
                else:
                    FN += 1
                pbar.update(1)
    else:
        for folder_path, img1, img2, in tqdm(image_pairs_list, desc="Processing input pairs", unit="pair"):
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
            

    # Calculate final FNR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    mean_FNR = 1 - TPR
    results["mean_FNR"] = mean_FNR

    print(f'Mean FNR across all IDs in group {main_dir}: {mean_FNR:.2%}')
    
    return results


