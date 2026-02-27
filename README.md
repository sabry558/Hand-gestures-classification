# ✋ Hand Gestures Classification

A machine learning project for classifying **18 hand gestures** using **MediaPipe hand landmarks**. Multiple models were trained, tuned via GridSearchCV, and tracked with MLflow to identify the best-performing classifier.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Gestures](#gestures)
- [Pipeline](#pipeline)
- [Model Comparison](#model-comparison)
- [Best Model](#best-model)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)

---

## 🔍 Overview

This project classifies hand gestures from 3D hand landmark coordinates (x, y, z for 21 keypoints) extracted using MediaPipe. The landmarks are preprocessed by centering on the wrist and normalizing by the distance to the middle finger base, making the features translation- and scale-invariant.

- **Dataset**: 25,675 samples across 18 gesture classes
- **Features**: 63 (21 landmarks × 3 coordinates)
- **Train/Test Split**: 80/20, stratified
- **Cross-Validation**: 3-fold Stratified K-Fold
- **Hyperparameter Tuning**: GridSearchCV
- **Experiment Tracking**: MLflow

---

## 🤚 Gestures

The model classifies the following **18 gestures**:

| | | | |
|---|---|---|---|
| call | dislike | fist | four |
| like | mute | ok | one |
| palm | peace | peace_inverted | rock |
| stop | stop_inverted | three | three2 |
| two_up | two_up_inverted | | |

---

## ⚙️ Pipeline

```
Raw CSV Data
    │
    ▼
Preprocessing (center on wrist, normalize by palm length)
    │
    ▼
Label Encoding
    │
    ▼
Train/Test Split (80/20, stratified)
    │
    ▼
GridSearchCV with 3-Fold Stratified CV
    │
    ▼
Evaluate on Test Set
    │
    ▼
Log to MLflow (params, metrics, model, confusion matrix)
```

---

## 📊 Model Comparison

All models were tuned with GridSearchCV (3-fold Stratified CV) and evaluated on the **held-out 20% test set** (5,135 samples).

| # | Model | Accuracy | Precision | Recall | F1-Score | Best Hyperparameters |
|:-:|:------|:--------:|:---------:|:------:|:--------:|:---------------------|
| 🥇 | **SVM (RBF)** | **99.0%** | **99.0%** | **99.0%** | **99.0%** | C=100, gamma=0.5, kernel=rbf |
| 🥈 | **Stacking Classifier** | **98.9%** | **98.9%** | **98.9%** | **98.9%** | SVM + XGB + RF + KNN → LR meta |
| 🥉 | **Voting Classifier** | **98.8%** | **98.7%** | **98.7%** | **98.7%** | SVM + XGB + RF + KNN (soft) |
| 4 | XGBoost | 98.5% | 98.5% | 98.5% | 98.5% | lr=0.2, max_depth=3, n_est=400 |
| 5 | KNN | 97.9% | 97.9% | 97.9% | 97.9% | k=3, weights=distance, euclidean |
| 6 | Random Forest | 97.9% | 97.9% | 97.9% | 97.9% | max_depth=None, n_est=300 |
| 7 | Decision Tree | 96.2% | 96.1% | 96.2% | 96.2% | entropy, max_depth=None, leaf=4, split=10 |
| 8 | Logistic Regression | 90.3% | 90.7% | 90.6% | 90.6% | C=100, max_iter=10000 |

> **Precision, Recall, and F1-Score** are **macro-averaged** across all 18 classes.

---

## 🏆 Best Model

### SVM with RBF Kernel — 99.0% Accuracy

The **Support Vector Machine** with an RBF kernel achieved the highest accuracy of **99.0%**, outperforming even the ensemble methods (Stacking and Voting classifiers).

**Best Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Kernel | RBF |
| C (Regularization) | 100 |
| Gamma | 0.5 |

**Per-Class Performance (Top highlights):**

| Gesture | Precision | Recall | F1-Score |
|---------|:---------:|:------:|:--------:|
| fist | 100.0% | 100.0% | 100.0% |
| dislike | 99.6% | 100.0% | 99.8% |
| two_up_inverted | 100.0% | 99.3% | 99.6% |
| three2 | 100.0% | 99.1% | 99.5% |
| peace_inverted | 99.7% | 99.3% | 99.5% |

> All 18 classes achieve **F1-scores above 96%**, demonstrating strong performance across every gesture.

---

## 📁 Project Structure

```
Hand-gestures-classification/
├── train_models.ipynb          # Main notebook: train, tune & compare all models
├── data_preprocessing.py       # Landmark normalization (center + scale)
├── train_model_helper.py       # GridSearchCV training & evaluation utilities
├── mlflow_helper.py            # MLflow logging helpers
├── hand_landmarks_data.csv     # Dataset (21 landmarks × 3 coords + label)
├── label_encoder.pkl           # Fitted LabelEncoder
├── artifacts/                  # Confusion matrix plots for each model
│   ├── svm_confusion_matrix.png
│   ├── stacking_classifier_confusion_matrix.png
│   ├── voting_classifier_confusion_matrix.png
│   ├── xgboost_confusion_matrix.png
│   ├── knn_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   └── logistic_regression_confusion_matrix.png
├── mlruns/                     # MLflow experiment tracking data
├── mlflow.db                   # MLflow SQLite backend store
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Hand-gestures-classification.git
cd Hand-gestures-classification

# Create and activate virtual environment
python -m venv env311
env311\Scripts\activate        # Windows
# source env311/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Training

Open `train_models.ipynb` in Jupyter and run all cells. Models are automatically logged to MLflow.

### View MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **scikit-learn** | ML models, GridSearchCV, metrics |
| **XGBoost** | Gradient boosting classifier |
| **MLflow** | Experiment tracking & model registry |
| **pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Confusion matrix visualization |
| **MediaPipe** | Hand landmark extraction (upstream) |
