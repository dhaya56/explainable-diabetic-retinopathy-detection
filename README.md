# 🩺 Explainable Diabetic Retinopathy Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-green)](https://www.kaggle.com/code/eminkorkut/diabetic-retinopathy-with-cnn/input)

A real-world Explainable AI system for **Diabetic Retinopathy classification** using a custom CNN architecture and **Grad-CAM explainability**, wrapped in an intuitive **Streamlit web app** with clinical guidance based on predictions.

---

## 🔍 Problem Statement

Diabetic Retinopathy (DR) is a leading cause of vision impairment globally. Early detection via retinal imaging and machine learning can empower patients and clinicians with actionable insights.

This project:
- Trains a CNN classifier on retinal images,
- Applies explainability (Grad-CAM) to highlight disease areas,
- Deploys a Streamlit app for real-time inference with guided clinical interpretation.

---

## 📦 Dataset

- **Source:** Kaggle Diabetic Retinopathy dataset  
  https://www.kaggle.com/code/eminkorkut/diabetic-retinopathy-with-cnn/input
- **Classes (5):**
  - No_DR
  - Mild
  - Moderate
  - Severe
  - Proliferative_DR
- **Total Images:** 3,554
- **Train/Test Split:** 80/20 (2,843 train, 711 test)
- **Input Size:** 512×512 RGB
- **Preprocessing:** Resize + normalization

> The raw dataset is not included in the repo (due to size). Instructions below explain how to prepare it locally.

---

## 🧠 Model Architecture

Custom Convolutional Neural Network:

- 4 convolution layers (32 → 256 channels)
- ReLU + MaxPool after each conv
- Dropout (0.25)
- Fully connected head (256 → 5 classes)
- Dynamic flattening using a mock tensor

The architecture is lightweight yet achieves strong performance without transfer learning.

---

## 📈 Training Details

| Component | Configuration |
|-----------|---------------|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr = 1e-3) |
| Epochs | 10 |
| Batch Size | 16 |
| Performance | **98.31% Test Accuracy** |
| Test Loss | 0.1716 |

The model achieves robust generalization with strong classification accuracy on held-out data.

---

## 🧪 Explainability — Grad-CAM

Explainability is critical in medical AI. Two explainability pipelines were explored:

### 📌 Notebook Visualization
- `torchcam.methods.SmoothGradCAMpp`
- Threshold-based hotspot detection
- Contour extraction around activated regions
- Overlayed on original image for visual insight

### 📌 Streamlit App Implementation
- Manual Grad-CAM via forward/backward hooks
- Background masking for retinal ROI
- Heatmap thresholding
- Highlight most activated region with circles
- Blended overlay shown alongside original image

This enables clinicians and end-users to *visually interpret* why the model made a certain prediction.

---

## 🚀 Streamlit App Usage

The Streamlit app (`app.py`) provides:

✔ Dynamic image upload  
✔ Real-time predictions  
✔ Grad-CAM heatmap visualization  
✔ Affected region highlighting  
✔ Clinical guidance by severity level  
✔ Downloadable Grad-CAM overlay

Simply upload a retinal image and interact with the gradients to explore outcomes.

---

## 🛠 Setup & Installation

### 1) Clone the Repo

```bash
git clone https://github.com/<your-username>/diabetic-retinopathy-detection.git
```