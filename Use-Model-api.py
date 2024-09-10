from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
model = tf.keras.models.load_model('Abdal_XSS_AI_Engine.h5')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentences = data['sentences']

    # Preprocess the input data using the vectorizer
    X_new = vectorizer.transform(sentences).toarray()

    # Make predictions
    predictions = (model.predict(X_new) > 0.5).astype(int)

    # Prepare and return the response
    response = {
        'predictions': ['XSS Detected' if pred == 1 else 'No XSS Detected' for pred in predictions.flatten()]
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
