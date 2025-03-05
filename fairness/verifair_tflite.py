import os
import random
import itertools
from itertools import combinations
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import tensorflow.lite as tflite
from ultralytics import YOLO
import numpy as np
import cv2
from scipy.spatial.distance import cosine

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

LOG_FILE = "verification_results.yaml"

def log_results(dataset_dir, model_path, metric, value, total_pairs, num_selected, FP=None, FN=None):
    """ Logs results to a YAML file """
    log_entry = {
        "dataset": dataset_dir,
        "model_name": model_path,
        "metric": metric,
        "value": round(value, 4),
        "total_pairs": total_pairs,
        "num_selected": num_selected,
    }
    if FP is not None:
        log_entry["False Positives"] = FP
    if FN is not None:
        log_entry["False Negatives"] = FN
    
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            try:
                logs = yaml.safe_load(file) or []
            except yaml.YAMLError:
                pass

def preprocess_image(image_path):
    model = YOLO("yolo11n.pt")
    # Run inference
    results = model(image_path)
    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        clss = result.boxes.cls  # Class indices of detected object
    for cls in clss:
        if model.names[int(cls)] == 'person':
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (160,160))
            image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0
            return image
        if model.names[int(cls)] != 'person':
            return None 

def get_embedding(interpreter, image):
    """Generates embeddings from a TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index']).flatten()

def verify(model_path, img1_path, img2_path, threshold=0.5):
    """Compares two images using a TFLite model and returns verification result."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    img1, img2 = preprocess_image(img1_path), preprocess_image(img2_path)
    if img1 is None or img2 is None:
        return None  # Skip invalid images
    
    emb1, emb2 = get_embedding(interpreter, img1), get_embedding(interpreter, img2)
    similarity = 1 - cosine(emb1, emb2)
    
    return similarity > threshold

def verify_pair(args):
    """Wrapper function for parallel processing of image pairs."""
    img1, img2, model_path = args
    return verify(model_path, img1, img2)

def process_pairs(image_pairs, model_path, use_multiprocessing=False, num_cores=None):
    """Processes image pairs using TFLite model with optional multiprocessing."""
    valid_results = []

    if use_multiprocessing:
        if num_cores is None or num_cores < 1 or num_cores > cpu_count():
            raise ValueError(f"num_cores must be between 1 and {cpu_count()}")

        with Pool(num_cores) as pool:
            results = list(tqdm(pool.imap_unordered(verify_pair, [(pair[0], pair[1], model_path) for pair in image_pairs]), 
                                total=len(image_pairs), desc="Processing pairs", unit="pair"))
        valid_results = [r for r in results if r is not None]  # Remove failed pairs
    else:
        for pair in tqdm(image_pairs, desc="Processing pairs", unit="pair"):
            result = verify_pair((pair[0], pair[1], model_path))
            if result is not None:
                valid_results.append(result)
    
    return valid_results

def FPR_tflite(dataset_dir, model_path, percentage=100, use_multiprocessing=False, num_cores=None):
    """Calculates the False Positive Rate (FPR) for a TFLite model."""
    subfolders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])
    
    image_pairs = [(os.path.join(dataset_dir, f1, img1), os.path.join(dataset_dir, f2, img2))
                   for f1, f2 in itertools.combinations(subfolders, 2)
                   for img1 in os.listdir(os.path.join(dataset_dir, f1)) if os.path.splitext(img1)[1].lower() in SUPPORTED_EXTENSIONS
                   for img2 in os.listdir(os.path.join(dataset_dir, f2)) if os.path.splitext(img2)[1].lower() in SUPPORTED_EXTENSIONS]

    total_pairs = len(image_pairs)
    num_selected = int((percentage / 100) * total_pairs)
    image_pairs = random.sample(image_pairs, num_selected)

    results = process_pairs(image_pairs, model_path, use_multiprocessing, num_cores)
    num_processed = len(results)
    FP = sum(results) if results else 0  # Count True results (False Positives)
    FPR_value = FP / num_processed if num_processed > 0 else 0

    print(f'Total possible pairs: {total_pairs}')
    print(f'Processed pairs: {num_processed}')
    print(f'Mean FPR: {FPR_value:.4%}')
    log_results(dataset_dir, model_path, "FPR", FPR_value, total_pairs, num_processed, FP=FP)
    
    return FPR_value

def FNR_tflite(dataset_dir, model_path, use_multiprocessing=False, num_cores=None):
    """Calculates the False Negative Rate (FNR) for a TFLite model."""
    subfolders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])
    
    image_pairs = [(os.path.join(dataset_dir, folder, img1), os.path.join(dataset_dir, folder, img2))
                   for folder in subfolders
                   for img1, img2 in combinations(sorted(os.listdir(os.path.join(dataset_dir, folder))), 2)
                   if os.path.splitext(img1)[1].lower() in SUPPORTED_EXTENSIONS and os.path.splitext(img2)[1].lower() in SUPPORTED_EXTENSIONS]

    total_pairs = len(image_pairs)
    
    results = process_pairs(image_pairs, model_path, use_multiprocessing, num_cores)
    num_processed = len(results)
    FN = sum(not match for match in results) if results else 0  # Count False results (False Negatives)
    FNR_value = FN / num_processed if num_processed else 0

    print(f'Total possible pairs: {total_pairs}')
    print(f'Processed pairs: {num_processed}')
    print(f'Mean FNR: {FNR_value:.4%}')
    log_results(dataset_dir, model_path, "FNR", FNR_value, total_pairs, num_processed, FN=FN)
    
    return FNR_value
