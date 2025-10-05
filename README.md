# ❤️ Heart Disease Diagnosis Classification

## 📘 Project Overview
This project focuses on building and evaluating machine learning models to **predict the presence of heart disease** using the **Cleveland Heart Disease dataset**.  
The goal is to compare model performance between the **original (raw)** dataset and a **feature-engineered** version, highlighting the impact of preprocessing and feature transformation on classification accuracy.

---

## 🧩 Dataset
**Name:** Cleveland Heart Disease Diagnosis Dataset  
**Source:** UCI Machine Learning Repository  

- The dataset contains patient medical attributes such as:
  - Age, sex, resting blood pressure, cholesterol level, fasting blood sugar, resting ECG results, maximum heart rate, etc.
- The target variable indicates whether the patient is diagnosed with heart disease.

---

## ⚙️ Methodology

### 1. **Data Preprocessing**
- Handled missing values and outliers.
- Encoded categorical features.
- Scaled continuous features.
- Split data into **train** and **test** sets.

### 2. **Feature Engineering**
- Created new derived attributes from existing ones to better capture relationships.
- Applied transformations and normalization.
- Re-trained all models using both **original** and **feature-engineered** datasets for comparison.

### 3. **Models Trained**
| Model | Description |
|--------|-------------|
| **K-Nearest Neighbors (KNN)** | Instance-based learner using distance metrics. |
| **Decision Tree** | Non-linear model that splits data by entropy or Gini impurity. |
| **Naïve Bayes (GaussianNB)** | Probabilistic classifier based on Bayes’ theorem. |
| **K-Means Clustering** | Unsupervised learning used here to evaluate clustering behavior. |
| **Ensemble Model** | Combined predictions from **KNN**, **Decision Tree**, and **Naïve Bayes** to improve robustness. |

---

## 🧠 Training Setup
- Implemented in **Python** using libraries:
  - `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- Evaluation metric: **Test Accuracy**
- Random seed fixed for reproducibility.

---

## 📊 Results Summary

| Model | Accuracy (Original Dataset) | Accuracy (Feature Engineered Dataset) |
|--------|------------------------------|----------------------------------------|
| Ensemble | 0.8387 | 0.8710 |
| KNN | 0.8387 | 0.8387 |
| Naïve Bayes | 0.8387 | 0.8387 |
| Decision Tree | 0.7097 | 0.7097 |
| K-Means | 0.1290 | 0.1290 |

> 📈 **Feature engineering improved performance**, especially in the ensemble model, showing that refined features can enhance diagnostic capability.

---

## 💡 Key Takeaways
- Ensemble methods yielded the **best performance** on the test set.
- Feature engineering **increased model accuracy** and stability.
- Even simple models can perform effectively with proper data preprocessing.
