# Facial-Expression-Recognition
A high-performance, real-time emotion detection system built using Convolutional Neural Networks (CNNs) with live webcam integration. The model delivers **88% accuracy** and runs at **30 FPS**, optimized for dynamic, real-world facial expressions under varying conditions.

---

## 🧰 Tech Stack

- **Language**: Python  
- **Deep Learning**: TensorFlow, Keras  
- **Computer Vision**: OpenCV  
- **Scientific Computing**: NumPy  
- **Visualization**: Matplotlib  
- **IDE**: Visual Studio Code  

---

## 🚀 Key Features

- 🎥 **Live Webcam Integration**: Real-time facial expression recognition using OpenCV.
- 🧠 **CNN-Based Model**: Accurate classification across key emotion categories (happy, sad, angry, neutral, etc.).
- ⚡ **Optimized for Speed**: 30 FPS performance ensures smooth, real-time inference.
- 🌟 **Robust in Real Conditions**: Handles diverse lighting and facial variations.
- 🎯 **High Accuracy**: Achieved 88% testing accuracy with a 25% reduction in false positives via tuning.

---

## 📊 Emotions Detected

- 😄 Happy  
- 😢 Sad  
- 😠 Angry  
- 😐 Neutral  
- 😨 Fear  
- 😲 Surprise  

---

## 🖥️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Santhosh1026/emotion-detection
cd emotion-detection

🧪 Model Training Overview
Dataset: FER-2013

Model Architecture: Custom CNN with batch normalization and dropout

Preprocessing: Face detection (Haar cascade), grayscale conversion, normalization

Evaluation: Accuracy, confusion matrix, F1 score, real-time inference validation

emotion-detection/
├── model/
│   └── emotion_model.h5               # Trained CNN model file
│
├── utils/
│   ├── face_detector.xml              # Haarcascade for face detection
│   └── preprocessing.py               # Helper functions for preprocessing
│
├── main.py                            # Entry point: Loads model and starts real-time detection
├── train_model.py                     # Script to train and save the CNN model
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation

