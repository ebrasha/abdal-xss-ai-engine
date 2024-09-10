# Author: Ebrahim Shafiei (EbraSha)

import os
import tensorflow as tf
import pickle

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Set TensorFlow logging level to 'ERROR' to suppress the info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check if model and vectorizer files exist
model_path = 'Abdal_XSS_AI_Engine.keras'
vectorizer_path = 'vectorizer.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

# Load the model from the Keras format
model_name = "Abdal XSS AI Engine"
model = tf.keras.models.load_model(model_path)

# Load the vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Read new data (sentences) from a file (e.g., 'attack-xss-payload.txt')
input_file = 'attack-xss-payload.txt'
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

with open(input_file, 'r', encoding='utf-8') as file:
    new_sentences = [line.strip() for line in file if line.strip()]  # Reading each line from file

# Check if any sentence exists for prediction
if not new_sentences:
    raise ValueError("No data available for prediction.")

# Preprocess the new data using the loaded TF-IDF vectorizer
X_new = vectorizer.transform(new_sentences).toarray()

# Predict using the loaded model
predictions = (model.predict(X_new) > 0.5).astype(int)

# Print predictions
for i, sentence in enumerate(new_sentences):
    print(f"Sentence: {sentence}")
    print(f"Prediction: {'XSS Detected' if predictions[i] == 1 else 'No XSS Detected'}\n")
