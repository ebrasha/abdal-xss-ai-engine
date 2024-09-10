# Abdal XSS AI Engine

## 🎤 README Translation
- [English](README.md)
- [فارسی](README.fa.md)

 

<p align="center"><img src="scr.jpg?raw=true"></p>


## 💎 General purpose
The Abdal XSS AI Engine was developed to provide a free and advanced solution for combating XSS attacks, particularly in Iran, where there is a lack of local cybersecurity models. This AI-based model addresses the crucial need for enhanced cybersecurity and aims to protect users by preventing XSS attacks more effectively.


## 🛠️ Development Environment Setup
- **Python 3.7 or higher**
- **Flask** (for building RESTful APIs)
- **TensorFlow** (Deep Learning models)
- **Scikit-learn** (for text preprocessing and TF-IDF vectorization)
- **Pandas** (for large-scale data management)
- **Deep understanding of web security and XSS attacks**
- **Git** (version control and repository management)



### 🔥 Requirements

- **Python 3.7 or higher**
- **Flask**
- **TensorFlow**
- **Scikit-learn**
- **Pandas**
- **Pickle**


## ✨ Features

- Ability to process and detect hundreds of thousands of XSS patterns, including new emerging threats.
- Utilizes a deep learning model with multiple Dense and Dropout layers.
- Trained using a combined dataset from multiple CSV files.
- Employs TF-IDF technique for text feature extraction from XSS attacks.
- Capability to improve model accuracy with new data and continuous updates.
- Supports model optimization using the Adam optimizer and accuracy metrics.
- Saves the final model and vectorizer for future use and deployment in various environments.


## 📝️ How it Works?
 
You can use the model as an API for detecting XSS attacks by using the following code:
```python
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

```
In addition to the API, you can also use the model to read data from a text file and detect attacks. The following code is an example of this use case:

```python
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

```


## ❤️ Donation

https://ebrasha.com/abdal-donation

## 🤵 Programmer
Handcrafted with Passion by Ebrahim Shafiei (EbraSha)

E-Mail = Prof.Shafiei@Gmail.com

Telegram: https://t.me/ProfShafiei

## ☠️ Reporting Issues

If you are facing a configuration issue or something is not working as you expected to be, please use the **Prof.Shafiei@Gmail.com** . Issues on GitLab  or Github are also welcomed.


