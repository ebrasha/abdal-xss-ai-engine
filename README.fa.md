# Abdal XSS AI Engine

## 🎤 ترجمه اطلاعات نرم افزار
- [English](README.md)
- [فارسی](README.fa.md)

 
<p align="center"><img src="scr.jpg?raw=true"></p>


## 💎 هدف اصلی
مدل هوش مصنوعی  Abdal XSS AI Engine با هدف ارائه یک راهکار پیشرفته و رایگان برای مقابله با حملات XSS در ایران توسعه داده شده است. با توجه به نبود ابزارهای مناسب سایبری داخلی، این مدل به عنوان یک نیاز ضروری برای افزایش امنیت در فضای سایبری ایران طراحی شده تا از حملات XSS جلوگیری کند و کاربران ایرانی بتوانند از حفاظت بهتری برخوردار شوند.


## 🛠️ پیش نیاز برای برنامه نویسان
- **Python 3.7 یا بالاتر**
- **Flask** (برای ساخت RESTful API)
- **TensorFlow** (مدل‌های یادگیری عمیق)
- **Scikit-learn** (برای پیش‌پردازش داده‌های متنی و بردار سازی TF-IDF)
- **Pandas** (برای مدیریت داده‌ها در مقیاس بزرگ)
- **درک عمیق از امنیت وب و حملات XSS**
- **Git** (کنترل نسخه و مدیریت مخازن)


### 🔥 پیشنیازها

- **Python 3.7 یا بالاتر**
- **Flask**
- **TensorFlow**
- **Scikit-learn**
- **Pandas**
- **Pickle**


## ✨ قابلیت ها

- قابلیت پردازش صدها هزار الگوی XSS و شناسایی حملات جدید.
- استفاده از مدل یادگیری عمیق با چند لایه Dense و Dropout.
- آموزش مدل با استفاده از مجموعه داده ترکیبی از فایل‌های CSV.
- استفاده از تکنیک TF-IDF برای استخراج ویژگی‌های متنی از حملات XSS.
- قابلیت افزایش دقت مدل با داده‌های جدید و به‌روز رسانی مداوم.
- پشتیبانی از بهینه‌سازی مدل با استفاده از روش Adam و معیار دقت (accuracy).
- ذخیره مدل نهایی و وکتورایزر برای استفاده در آینده و استقرار در محیط‌های مختلف.


## 📝️ چگونه کار می کند ؟

با استفاده از کد زیر می‌توانید از مدل به صورت یک API برای شناسایی حملات XSS استفاده کنید

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
علاوه بر API، می‌توانید از مدل برای خواندن داده‌ها از یک فایل متنی و شناسایی حملات استفاده کنید. کد زیر نمونه‌ای از این استفاده است:

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
## ❤️ کمک به پروژه

https://alphajet.ir/abdal-donation

## 🤵 برنامه نویس
دست ساز با عشق توسط ابراهیم شفیعی (ابراشا)

E-Mail = Prof.Shafiei@Gmail.com

Telegram: https://t.me/ProfShafiei

## ☠️ گزارش خطا

اگر با مشکلی در پیکربندی مواجه هستید یا چیزی آنطور که انتظار دارید کار نمی‌کند، لطفا از Prof.Shafiei@Gmail.com استفاده کنید.طرح مشکلات بر روی  GitLab یا Github نیز پذیرفته می‌شوند.



