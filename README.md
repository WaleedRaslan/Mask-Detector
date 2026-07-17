# 😷 Face Mask Detection Dashboard

A real-time Face Mask Detection system built using **TensorFlow**, **Keras (MobileNetV2 Transfer Learning)**, **OpenCV**, **MediaPipe**, and **Streamlit**.

The application provides an interactive dashboard capable of detecting face masks from both live webcam streams and uploaded images while logging predictions for statistical analysis.

---

## ✨ Features

### 📷 Real-Time Webcam Detection
- Live face detection using **MediaPipe Face Detection**
- Real-time face mask classification using a trained **MobileNetV2** model
- Bounding boxes with prediction labels and confidence scores

### 🖼️ Image Detection
- Upload any image
- Automatically detect every face in the image
- Predict whether each detected person is wearing a face mask

### 📊 Analytics Dashboard
- Automatic logging of every prediction
- Interactive statistics generated using **Matplotlib**
- Detection history table
- Confidence score distribution
- Daily detection statistics
- Mask vs No Mask distribution

---

## 🧠 Model Details

- Architecture: **MobileNetV2 (Transfer Learning)**
- Framework: TensorFlow / Keras
- Input Size: **224 × 224 × 3**
- Output:
  - **Mask**
  - **No Mask**
- Activation Function: **Sigmoid**
- Binary Classification

> The model includes an internal **Rescaling** layer, so no manual normalization (`/255`) is required during inference.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- Streamlit

---

## 📁 Project Structure

```text
mask_dashboard/
│
├── app.py                 # Streamlit application
├── detector.py            # Detection pipeline
├── MaskDetector.keras     # Trained MobileNetV2 model
├── requirements.txt
└── detection_log.csv      # Automatically generated detection history
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
cd mask_dashboard
```

Create a virtual environment (optional):

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The dashboard will be available at:

```
http://localhost:8501
```

---

## 📊 Detection Logging

Every prediction is automatically stored in:

```
detection_log.csv
```

Each record contains:

- Timestamp
- Detection Source (Webcam / Uploaded Image)
- Prediction Label
- Confidence Score

These records are used to generate the dashboard statistics.

---

## 📈 Dashboard Analytics

The dashboard provides:

- Detection History
- Daily Detection Statistics
- Mask vs No Mask Distribution
- Confidence Score Distribution
- Source Filtering (Webcam / Uploaded Images)
- One-click Detection History Reset

---

## 🎯 Future Improvements

- Audio alert when a person is detected without a face mask
- Save detected face images alongside prediction results
- Replace CSV logging with SQLite or PostgreSQL
- Multi-camera support
- REST API deployment
- Docker support

---

## 👨‍💻 Author

Developed by **Waleed Raslan**

Computer Vision • Deep Learning • Artificial Intelligence
