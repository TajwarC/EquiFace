import os
import itertools
import random
from tqdm import tqdm
from deepface import DeepFace
import multiprocessing
from multiprocessing import Pool, cpu_count

def verify_pair(pair):
    """ Verifies if two images belong to the same person. """
    img1, img2 = pair
    result = DeepFace.verify(img1,
                             img2,
                             enforce_detection=False)
    return result["verified"]

def FPR(dataset_dir, percentage=100, use_multiprocessing=False, num_cores=None):
    """ Calculates the False Positive rate for a DeepFace model"""
    if use_multiprocessing:
        if num_cores is None:
            raise ValueError("You must specify the number of cores you want to use when use_multiprocessing=True")
        if num_cores < 1 or num_cores > cpu_count():
            raise ValueError(f"num_cores must be between 1 and {cpu_count()}")
        
    subfolders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])
    # Count images in each subfolder
    image_counts = {
        folder: len([img for img in os.listdir(os.path.join(dataset_dir, folder)) if img.endswith('.png')])
        for folder in subfolders
    }
    
    # Compute total number of possible inter-folder comparisons
    total_comparisons = sum(
        image_counts[folder1] * image_counts[folder2] 
        for folder1, folder2 in itertools.combinations(subfolders, 2)
    )
    
    print(f"Total number of input pairs: {total_comparisons}")
    
    # Generate all possible image pairs (excluding same subfolder comparisons)
    image_pairs = [
        (os.path.join(dataset_dir, folder1, img1), os.path.join(dataset_dir, folder2, img2))
        for folder1, folder2 in itertools.combinations(subfolders, 2)
        for img1 in os.listdir(os.path.join(dataset_dir, folder1)) if img1.endswith('.png')
        for img2 in os.listdir(os.path.join(dataset_dir, folder2)) if img2.endswith('.png')
    ]
    
    # Select a random subset based on the specified percentage
    num_selected = int((percentage / 100) * total_comparisons)
    image_pairs = random.sample(image_pairs, num_selected)
    
    print(f"Selected {num_selected} pairs for evaluation ({percentage}% of total)")
    
    FP = 0
    TN = 0
    
    if use_multiprocessing:
        # Progress bar with multiprocessing
        manager = multiprocessing.Manager()
        progress = manager.Value("i", 0)
        lock = manager.Lock()
        
        def update_progress(_):
            with lock:
                progress.value += 1
                pbar.update(1)
        
        with tqdm(total=num_selected, desc="Processing input pairs", unit="pair") as pbar:
            with Pool(processes=num_cores) as pool:
                results = pool.imap_unordered(verify_pair, image_pairs)
                
                for result in results:
                    if result:
                        FP += 1
                    else:
                        TN += 1
                    update_progress(result)
    else:
         with tqdm(total=num_selected, desc="Processing input pairs", unit="pair") as pbar:
            for pair in image_pairs:
                result = verify_pair(pair)
                if result:
                    FP += 1
                else:
                    TN += 1
                pbar.update(1)
        
    # Calculate metrics
    FPR = FP / num_selected if num_selected > 0 else 0
    TNR = TN / num_selected if num_selected > 0 else 0
    
    # Print results
    print(f"False Positives: {FP}")
    print(f"True Negatives: {TN}")
    print(f"False Positive Rate (FPR): {FPR:.2f}")
    print(f"True Negative Rate (TNR): {TNR:.2f}")
