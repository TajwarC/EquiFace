import tensorflow as tf
import tarfile

# Extract model from tar.gz file
# Downloaded from https://www.kaggle.com/models/faiqueali/facenet-tensorflow
model_path = "facenet-tensorflow-tensorflow2-default-v2.tar.gz"
extract_path = "facenet_model"

with tarfile.open(model_path, "r:gz") as tar:
    tar.extractall(path=extract_path)
    
model_dir = extract_path 
model = tf.saved_model.load(model_dir)

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
