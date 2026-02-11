# 🩺 Explainable Diabetic Retinopathy Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-orange)](https://streamlit.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-green)](https://www.kaggle.com/code/eminkorkut/diabetic-retinopathy-with-cnn/input)

An end-to-end **Explainable AI system** for multi-class Diabetic Retinopathy (DR) classification using a custom Convolutional Neural Network (CNN) and Grad-CAM visualization, deployed through a Streamlit web application for real-time inference and interpretability.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Model Performance](#model-performance)
- [Explainability with Grad-CAM](#explainability-with-grad-cam)
- [Streamlit Deployment](#streamlit-deployment)
- [Clinical Decision Support Layer](#clinical-decision-support-layer)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Sample Screenshots](#sample-screenshots)
- [Setup & Installation](#setup--installation)
- [Pretrained Weights](#pretrained-weights)

---

## Project Overview

Diabetic Retinopathy (DR) is a diabetes-related complication that affects the retina and can lead to vision loss if undetected.

This project builds:

- A custom CNN classifier for 5 DR severity stages  
- A Grad-CAM explainability pipeline  
- A real-time interactive Streamlit application  
- A clinical guidance layer based on prediction severity  

The system emphasizes **interpretability, robustness, and deployment readiness**, not just raw accuracy.

---

## Dataset Description

- **Source:** Kaggle Diabetic Retinopathy Dataset  
  https://www.kaggle.com/code/eminkorkut/diabetic-retinopathy-with-cnn/input

- **Total Images:** 3,554 retinal images  
- **Classes (5):**
  - No_DR
  - Mild
  - Moderate
  - Severe
  - Proliferate_DR

- **Split:** 80% Training / 20% Testing  
  - Train: 2,843  
  - Test: 711  

- **Preprocessing:**
  - Resize to 512×512  
  - Convert to tensor  

The dataset is not included in this repository due to size constraints.

---

## Model Architecture

Custom CNN implemented in PyTorch:

- 4 Convolutional Layers:
  - 3 → 32  
  - 32 → 64  
  - 64 → 128  
  - 128 → 256  
- ReLU activation  
- MaxPooling after each convolution  
- Dropout (0.25)  
- Fully connected head:
  - Flatten → 256 → 5 classes  

The flatten dimension is computed dynamically to maintain architectural flexibility.

---

## Training Pipeline

- Loss Function: `CrossEntropyLoss`  
- Optimizer: `Adam`  
- Learning Rate: `1e-3`  
- Epochs: 10  
- Batch Size: 16  

Training flow:

1. Load dataset using `ImageFolder`
2. Perform 80/20 train-test split
3. Train for 10 epochs
4. Evaluate on unseen test data
5. Save model weights and class metadata

---

## Model Performance

| Metric | Value |
|--------|--------|
| Test Accuracy | **98.31%** |
| Test Loss | 0.1716 |

The model achieves strong generalization despite not using transfer learning.

---

## Explainability with Grad-CAM

Medical AI systems require transparency.

This project includes two Grad-CAM implementations:

### Notebook-Based Explainability
- SmoothGradCAM++ (torchcam)
- Threshold-based hotspot detection
- Contour extraction
- Region highlighting

### Streamlit App Grad-CAM
- Manual forward/backward hooks
- Gradient-weighted feature maps
- Gaussian smoothing
- Background masking
- Activation threshold filtering
- Region localization and overlay blending

This allows visual interpretation of model decisions.

---

## Streamlit Deployment

The `app.py` provides:

- Image upload interface  
- Real-time prediction  
- Grad-CAM heatmap visualization  
- Adjustable transparency  
- Activation threshold control  
- Region highlighting  
- Downloadable overlay image  

Run locally:

```bash
streamlit run app.py
```