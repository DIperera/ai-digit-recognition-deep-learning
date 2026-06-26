# 🧠 Digit Predictor - Deep Learning Project (MNIST)

A complete end-to-end **Deep Learning project** that recognizes handwritten digits (0–9) using a trained neural network model. The project was trained and tested using **Google Colab + TensorFlow + Gradio**, and later deployed as a **Flask web application** for real-world usage.

---

## 🚀 Project Overview

This project demonstrates a full machine learning pipeline:

- Dataset: **MNIST Handwritten Digits Dataset**
- Model: **Neural Network (TensorFlow/Keras)**
- Training Environment: **Google Colab**
- Initial Testing UI: **Gradio**
- Deployment: **Flask Web Application**

The model predicts handwritten digits from uploaded images.

---

## 🧠 Workflow

### 1️⃣ Model Training (Google Colab)

- Loaded MNIST dataset using TensorFlow
- Normalized and preprocessed data
- Built a simple neural network:
  - Flatten layer
  - Dense hidden layer (ReLU)
  - Output layer (Softmax)
- Trained the model for multiple epochs
- Achieved high accuracy on test data

### 2️⃣ Model Testing (Gradio Interface)

- Built a quick **Gradio UI** in Colab
- Tested model with handwritten digit images
- Verified predictions and performance
- Saved trained model as `.h5` file

### 3️⃣ Model Deployment (Flask Application)

After downloading the trained model, a Flask application was developed:

- Backend: Flask (Python)
- Frontend: HTML + CSS + JavaScript
- Model Loading: TensorFlow Keras
- Input: Image upload (handwritten digit)
- Output: Predicted digit (0–9)

---

## 🖥️ Features

- 📤 Upload handwritten digit images
- 🤖 Real-time prediction using trained model
- 🎯 High accuracy digit classification
- 🎨 Clean interactive UI (HTML + CSS)
- ⚡ Lightweight Flask backend
- 🧠 Deep learning powered inference

---

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / Keras 🤖
- NumPy
- Flask 🌐
- HTML / CSS / JavaScript
- Google Colab
- Gradio (for testing)

---

## 📁 Project Structure

```text
ai-digit-recognition-deep-learning/
│
├── app.py
├── mnist_model.h5
├── README.md
├── requirements.txt
├── Colab/
│   ├── gradio_app.py
│   ├── model_load.py
│   └── train_colab.py
├── static/
│   └── style.css
└── templates/
	└── index.html
```

---

## ▶️ How to Run Locally

### 1. Install dependencies

```bash
pip install flask tensorflow numpy pillow
```

### 2. Run Flask app

```bash
python app.py
```

### 3. Open in browser

```text
http://127.0.0.1:5000/
```

---

## 📊 Model Details

- Input Shape: 28 × 28 grayscale images
- Output Classes: 10 digits (0–9)
- Activation: ReLU + Softmax
- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam

---

## 🎯 Example Use Case

1. Upload a handwritten digit image (0–9)
2. System processes and normalizes image
3. Model predicts digit
4. Result is displayed instantly on web UI

---

## 📌 Key Learning Outcomes

- End-to-end ML pipeline development
- Model training in Google Colab
- Using Gradio for rapid testing
- Flask deployment of ML model
- Image preprocessing for real-world inputs
- Frontend + backend integration

---

## 🚀 Future Improvements

- Improve model using CNN architecture
- Add confidence score (probability output)
- Real-time drawing canvas input
- Deploy on cloud (Azure / Render / AWS)
- Mobile responsive UI improvements
