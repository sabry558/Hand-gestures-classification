# Hand Gestures Classification

A comprehensive machine learning project for classifying hand gestures using 3D hand landmark data extracted via [MediaPipe](https://developers.google.com/mediapipe).

## Overview

This project reads raw structural coordinates of hands (x, y, z for 21 structural landmarks) to classify various static hand gestures (e.g., "call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "rock", "stop", etc.). The dataset consists of 25,675 samples with 63 numerical features and one target label column. The end goal is a real-time webcam inference system capable of translating physical hand gestures into text overlay.

## Project Structure

```text
Hand-gestures-classification/
├── Hand gestures.ipynb      # Main Jupyter Notebook with the full ML pipeline
├── hand_landmarks_data.csv  # Dataset with 21 3D hand landmark coordinates
├── model/
│   └── hand_gesture_svm_model.pkl  # Trained SVM model (best performing)
├── requirements.txt         # Python dependencies needed to run the project
├── .gitignore               # Git ignore rules
└── README.md                # This documentation file
```

## Detailed Pipeline

The `Hand gestures.ipynb` notebook implements the entire pipeline step-by-step:

### 1. Exploratory Data Analysis (EDA)
- **Data Loading**: Loads `hand_landmarks_data.csv` using Pandas.
- **Data Quality Check**: Verified there are no null values and no duplicate records in the dataset. All 63 coordinate features are cleanly formatted as `float64`.
- **Target Distribution**: Analyzed class distribution using bar charts to view the counts of each gesture sample.

### 2. Visualization
- Plotted the 21 3D hand landmarks in 2D space using Matplotlib scatter plots to physically verify the coordinate representation of different gestures.

### 3. Normalization & Preprocessing
To make the model robust against different hand sizes and distances from the camera, the data is heavily normalized:
- **Translation / Position Invariance**: Subtracted the wrist coordinate (Point 0) from all other 20 points, effectively making the wrist the absolute origin (0,0,0).
- **Scale Invariance**: Divided all coordinate distances by the Euclidean distance between Point 0 (wrist) and Point 12 (middle fingertip), creating uniform scaling regardless of how close the hand is to the camera.
- **Label Encoding**: Transformed categorical string labels into machine-readable integers using `LabelEncoder`.
- **Stratified Split**: Split the data into 80% training and 20% testing sets using `train_test_split(..., stratify=y)` to preserve class balances.

### 4. Model Training & Hyperparameter Tuning
We trained multiple models, utilizing `GridSearchCV` and `StratifiedKFold` to find the best hyperparameters for each algorithm:
- **K-Nearest Neighbors (KNN)**: Tuned `n_neighbors`, weights (uniform/distance), and metrics (euclidean/manhattan).
- **Logistic Regression**: Tuned regularization strength `C`.
- **Support Vector Machine (SVC)**: Tuned `C`, `gamma`, and `kernel` (rbf). Set `probability=True` for soft voting downstream.
- **Decision Tree**: Tuned `criterion`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **XGBoost**: Gradient boosted trees tuned for `n_estimators`, `learning_rate`, and `max_depth`.
- **Random Forest**: Ensembled trees tuned for `n_estimators` and `max_depth`.

### 5. Ensemble Classifiers
To push performance even higher, we combined the top-tier models (SVM, XGBoost, Random Forest, KNN):
- **Voting Classifier**: Aggregated predictions via soft voting.
- **Stacking Classifier**: Trained a meta-model on the predictions of the base estimators.

### 6. Model Evaluation & Comparison

Using metrics like `accuracy`, `precision`, `recall`, and `f1_score`, the models were systematically compared on the **held-out 20% test set** (5,135 samples).

#### 📊 Model Comparison Table

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

> **Precision, Recall, and F1-Score** are **macro-averaged** across all 18 gesture classes.

#### 🏆 Best Model: Support Vector Machine (SVC with RBF Kernel)

The **SVM (RBF kernel)** achieved the highest overall performance with **99.0% accuracy** on the test set, outperforming even ensemble methods (Stacking and Voting classifiers). Key details:

- **Best Hyperparameters**: `C=100`, `gamma=0.5`, `kernel=rbf`, `probability=True`
- **Test Accuracy**: 99.0%
- **Macro-Averaged Precision**: 99.0%
- **Macro-Averaged Recall**: 99.0%
- **Macro-Averaged F1-Score**: 99.0%

The trained SVM model is saved as `model/hand_gesture_svm_model.pkl` for deployment in the real-time prediction system.

### 7. Real-time Prediction System
The final section of the project connects the winning SVM model to a live webcam feed using OpenCV and MediaPipe Hands:
- Captures video frames in real-time.
- Extracts hand landmarks using `mp.solutions.hands`.
- Applies the exact same normalization (translating wrist to origin, scaling by wrist-to-middle-finger distance).
- Predicts the gesture using the trained model.
- Overlays the predicted gesture text directly onto the video feed.

## Requirements & Installation

- Python 3.11+ is recommended.

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependencies Included:
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Computer Vision**: `opencv-python`, `mediapipe`

## Usage

1. Clone the repository and navigate to the project root.
2. Install the necessary packages.
3. Open `Hand gestures.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells sequentially to observe data processing, model training, and evaluation.
5. In the final cell, your webcam will activate. Show different hand gestures to the camera to see the real-time classification overlay! Press `q` to exit the webcam view.
