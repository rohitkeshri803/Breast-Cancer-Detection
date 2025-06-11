
# Breast Cancer Detection using Neural Networks (PyTorch)

This project focuses on building a deep learning model to predict whether a tumor is malignant or benign using the Breast Cancer Wisconsin dataset. The goal is to demonstrate a complete ML pipeline — from data preprocessing to model training, evaluation, and tuning — using PyTorch.

---

## 📌 Problem Statement

Early diagnosis of breast cancer can save lives. Using a dataset of tumor features collected from digitized images of breast mass, this model predicts whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)**.

---

## 📊 Dataset

- **Source**: `sklearn.datasets.load_breast_cancer`
- **Samples**: 569
- **Features**: 30 numerical features (e.g., radius, texture, smoothness, etc.)
- **Target**: Binary classification (0 = malignant, 1 = benign)

---

## 🧠 Model Architecture

A simple **Feedforward Neural Network** built using PyTorch:
- **Input Layer**: 30 neurons
- **Hidden Layer**: 32 or 64 neurons (tuned via grid search)
- **Activation**: ReLU
- **Output Layer**: 1 neuron with Sigmoid (through `BCEWithLogitsLoss`)

---

## 🛠️ Features & Techniques

✅ Data Standardization (using `StandardScaler`)  
✅ GPU support (CUDA)  
✅ Proper train/val/test split  
✅ Early stopping based on validation loss  
✅ Cross-validation (5-fold)  
✅ Grid search over `hidden_size` and `learning_rate`  
✅ Classification report and confusion matrix  
✅ Final model saved with `.pth` file  
✅ Accuracy consistently >97% on test set  
✅ Clean plots for loss and accuracy over epochs

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~97–99% |
| Precision | 0.98 |
| Recall | 0.97 |
| F1 Score | 0.975 |

---

## 📂 Project Structure

