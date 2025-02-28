import os
from itertools import combinations
from tqdm import tqdm
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
    """Function to verify a pair of images and return result."""
    model_path, folder_path, img1, img2 = args
    result = verify(
        model_path,
        img1_path=os.path.join(folder_path, img1),
        img2_path=os.path.join(folder_path, img2),
    )
    return result['verified']

def FNR_tflite(dataset_dir, model_path, use_multiprocessing=False, num_cores=None):
    
    if use_multiprocessing:
        if num_cores is None:
            raise ValueError("You must specify the number of cores you want to use when use_multiprocessing=True")
        if num_cores < 1 or num_cores > cpu_count():
            raise ValueError(f"num_cores must be between 1 and {cpu_count()}")
    # Initiate results
    results = {}
    main_dir = dataset_dir
    individual_dirs = [f.name for f in os.scandir(main_dir) if f.is_dir()]
    
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
    
    if use_multiprocessing:
        with Pool(num_cores) as pool, tqdm(total=total_pairs, desc="Processing input pairs", unit="pair") as pbar:
            for is_match in pool.imap_unordered(verify_pair, image_pairs_list):
                if is_match:
                    TP += 1
                else:
                    FN += 1
                pbar.update(1)
    else:
        for folder_path, img1, img2 in tqdm(image_pairs_list, desc="Processing input pairs", unit="pair"):
            result = verify(
                model_path=model_path,
                img1_path=os.path.join(folder_path, img1),
                img2_path=os.path.join(folder_path, img2),
            )
            is_match = result['verified']
            if is_match:
                TP += 1
            else:
                FN += 1
    
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    mean_FNR = 1 - TPR
    results["mean_FNR"] = mean_FNR
    
    print(f'False Negatives: {FN}')
    print(f'Mean FNR across all IDs in group {main_dir}: {mean_FNR:.4%}')
    
    return results
