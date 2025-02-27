import os
import random
import itertools
import yaml
from tqdm import tqdm
from deepface import DeepFace
from multiprocessing import Pool, cpu_count

LOG_FILE = "verification_results.yaml"

# File types for images
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def verify_pair(args):
    """ Verifies if two images belong to the same person using the specified DeepFace model. """
    img1, img2, model_name = args
    try:
        result = DeepFace.verify(img1, img2, model_name=model_name, 
                                 enforce_detection=True, detector_backend='yolov8')
        return result["verified"]
    except ValueError:
        return None  # Skip input pair if detection backend fails to find a face

def log_results(dataset_dir, model_name, metric, value, total_pairs, num_selected, FP=None, FN=None):
    """ Logs results to a YAML file """
    log_entry = {
        "dataset": dataset_dir,
        "model_name": model_name,
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
    
    logs.append(log_entry)
    with open(LOG_FILE, "w") as file:
        yaml.safe_dump(logs, file, default_flow_style=False)
    
    print(f"Logged {metric} result: {value:.4%} in {LOG_FILE}")

def process_pairs(image_pairs, model_name, use_multiprocessing=False, num_cores=None):
    """ Processes image pairs through model, using single or multiple cores. """
    valid_results = []
    if use_multiprocessing:
        if num_cores is None or num_cores < 1 or num_cores > cpu_count():
            raise ValueError(f"num_cores must be between 1 and {cpu_count()}")
        
        with Pool(num_cores) as pool:
            results = list(tqdm(pool.imap_unordered(verify_pair, [(pair[0], pair[1], model_name) for pair in image_pairs]), 
                                total=len(image_pairs), desc="Processing pairs", unit="pair"))
        valid_results = [r for r in results if r is not None]  # Remove failed pairs
    else:
        for pair in tqdm(image_pairs, desc="Processing pairs", unit="pair"):
            result = verify_pair((pair[0], pair[1], model_name))
            if result is not None:
                valid_results.append(result)
    
    return valid_results

def FPR(dataset_dir, model_name="VGG-Face", percentage=100, use_multiprocessing=False, num_cores=None):
    """ Calculates False Positive Rate (FPR). """
    subfolders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])
    image_pairs = [(os.path.join(dataset_dir, f1, img1), os.path.join(dataset_dir, f2, img2))
                   for f1, f2 in itertools.combinations(subfolders, 2)
                   for img1 in os.listdir(os.path.join(dataset_dir, f1)) if os.path.splitext(img1)[1].lower() in SUPPORTED_EXTENSIONS
                   for img2 in os.listdir(os.path.join(dataset_dir, f2)) if os.path.splitext(img2)[1].lower() in SUPPORTED_EXTENSIONS]
    
    total_pairs = len(image_pairs)
    num_selected = int((percentage / 100) * total_pairs)
    image_pairs = random.sample(image_pairs, num_selected)
    
    results = process_pairs(image_pairs, model_name, use_multiprocessing, num_cores)
    num_processed = len(results)
    FP = sum(results) if results else 0  # Count True results (False Positives)
    FPR_value = FP / num_processed if num_processed > 0 else 0
    
    print(f'Total possible pairs: {total_pairs}')
    print(f'Processed pairs: {num_processed}')
    print(f'Mean FPR: {FPR_value:.4%}')
    log_results(dataset_dir, model_name, "FPR", FPR_value, total_pairs, num_processed, FP=FP)
    return FPR_value

def FNR(dataset_dir, model_name="VGG-Face", use_multiprocessing=False, num_cores=None):
    """ Calculates False Negative Rate (FNR). """
    subfolders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    image_pairs = [(os.path.join(dataset_dir, folder, img1), os.path.join(dataset_dir, folder, img2))
                   for folder in subfolders
                   for img1, img2 in itertools.combinations(sorted(os.listdir(os.path.join(dataset_dir, folder))), 2)
                   if os.path.splitext(img1)[1].lower() in SUPPORTED_EXTENSIONS and os.path.splitext(img2)[1].lower() in SUPPORTED_EXTENSIONS]
    
    total_pairs = len(image_pairs)
    num_selected = total_pairs 
    
    results = process_pairs(image_pairs, model_name, use_multiprocessing, num_cores)
    num_processed = len(results)
    FN = sum(not match for match in results) if results else 0  # Count False results (False Negatives)
    FNR_value = FN / num_processed if num_processed else 0
    
    print(f'Total possible pairs: {total_pairs}')
    print(f'Processed pairs: {num_processed}')
    print(f'Mean FNR: {FNR_value:.4%}')
    log_results(dataset_dir, model_name, "FNR", FNR_value, total_pairs, num_processed, FN=FN)
    return FNR_value
