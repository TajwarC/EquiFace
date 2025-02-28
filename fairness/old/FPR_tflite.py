import os
import itertools
import random
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
import tensorflow.lite as tflite
import numpy as np
import cv2
from scipy.spatial.distance import cosine

def preprocess_image(image_path, target_size=(160, 160)):
    """Load and preprocess the image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0) 
    image = image.astype(np.float32) / 255.0 
    return image

def get_embedding(interpreter, image):
    """Generate embedding from the image using the TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])
    return embedding.flatten()

def verify(model_path, img1_path, img2_path, threshold=0.5):
    """Compare two images and verify if they belong to the same person."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)
    
    emb1 = get_embedding(interpreter, img1)
    emb2 = get_embedding(interpreter, img2)
    
    similarity = 1 - cosine(emb1, emb2)
    
    return {'verified': similarity > threshold}


def verify_pair(args):
    """ Verifies if two images belong to the same person. """
    model_path, img1, img2 = args
    result = verify(model_path, img1, img2)
    return result["verified"]

def FPR_tflite(dataset_dir, model_path, percentage=100, use_multiprocessing=False, num_cores=None):
    """ Calculates the False Positive rate for a tflite model"""
    
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
        (model_path, os.path.join(dataset_dir, folder1, img1), os.path.join(dataset_dir, folder2, img2))
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
    print(f'Mean FPR across all IDs in group {dataset_dir}: {FPR:.4%}')
    
