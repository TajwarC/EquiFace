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
