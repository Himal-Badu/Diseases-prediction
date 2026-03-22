# 🫀 Heart Disease Prediction using Machine Learning

A machine learning toolkit that predicts **thalassemia types** (fixed defect, normal, reversible defect) from clinical heart health data using multiple classification algorithms. Built with Python and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

Heart disease remains one of the leading causes of death worldwide. Early prediction using clinical data can assist healthcare professionals in diagnosis and treatment planning. This project applies three classification models to a preprocessed heart disease dataset and compares their performance.

### Key Features

- **Multi-class classification** — Predicts thalassemia type: `fixed defect`, `normal`, or `reversible defect`
- **Three ML models** — Logistic Regression, Decision Tree, and Random Forest
- **Pre-trained models** — Ready-to-use `.pkl` files for instant predictions
- **Feature scaling** — Includes a saved scaler for consistent data preprocessing
- **Comprehensive evaluation** — Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics

---

## 📊 Dataset

The project uses a **preprocessed heart disease dataset** with one-hot encoded categorical features.

| Attribute | Detail |
|-----------|--------|
| **Features** | Age, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope, number of major vessels, sex, chest pain type |
| **Target** | Thalassemia type — `0: fixed defect`, `1: normal`, `2: reversible defect` |
| **Encoding** | One-hot encoding applied to `sex` and `cp` (chest pain) |
| **Total Features** | 16 |

---

## 🤖 Models & Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.00 | 0.00 | 0.00 | 0.00 |
| Decision Tree | 0.00 | 0.00 | 0.00 | 0.00 |
| **Random Forest** 🏆 | **1.00** | **1.00** | **1.00** | **1.00** |

> **Best Model:** Random Forest with 100% accuracy on the test set.
>
> ⚠️ *Note: The dataset used is small (5 samples). These results demonstrate the pipeline — for production use, train on a larger dataset (e.g., the full [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)) for more reliable metrics.*

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Clone the Repository

```bash
git clone https://github.com/Himal-Badu/Diseases-prediction.git
cd Diseases-prediction
```

### Make Predictions

```python
import joblib
import numpy as np
import json

# Load the pre-trained model and scaler
model = joblib.load('disease_prediction_result/heart_random_forest_model.pkl')
scaler = joblib.load('disease_prediction_result/heart_scaler.pkl')

# Load feature names
with open('disease_prediction_result/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Example patient data
new_data = np.array([[
    58,    # age
    130,   # trestbps (resting blood pressure)
    220,   # chol (cholesterol)
    1,     # fbs (fasting blood sugar > 120 mg/dl)
    0,     # restecg (resting ECG results)
    150,   # thalch (max heart rate achieved)
    0,     # exang (exercise-induced angina)
    1.4,   # oldpeak (ST depression)
    0,     # slope (slope of peak exercise ST segment)
    0,     # ca (number of major vessels)
    0,     # sex_Female
    1,     # sex_Male
    0,     # cp_asymptomatic
    0,     # cp_atypical angina
    0,     # cp_non-anginal
    1      # cp_typical angina
]])

# Scale and predict
scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)

target_map = {0: "Fixed Defect", 1: "Normal", 2: "Reversible Defect"}
print(f"Prediction: {target_map[prediction[0]]}")
```

---

## 📁 Project Structure

```
Diseases-prediction/
├── README.md
└── disease_prediction_result/
    ├── feature_names.json              # Feature names in correct order
    ├── heart_decision_tree_model.pkl   # Trained Decision Tree model
    ├── heart_logistic_regression_model.pkl  # Trained Logistic Regression model
    ├── heart_random_forest_model.pkl   # Trained Random Forest model (best)
    ├── heart_scaler.pkl                # StandardScaler for preprocessing
    ├── heart_results.json              # Evaluation metrics for all models
    └── summary_report.json             # Dataset summary & best model info
```

---

## 🔮 Future Improvements

- [ ] Train on the full UCI Heart Disease dataset (303+ samples)
- [ ] Add cross-validation for more robust evaluation
- [ ] Implement hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- [ ] Build a web interface using Streamlit or Flask
- [ ] Add data visualization (feature importance, confusion matrices, ROC curves)
- [ ] Support additional disease prediction (diabetes, kidney disease, etc.)

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **ML Framework:** scikit-learn
- **Data Processing:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Model Serialization:** joblib

---

## 👤 Author

**Himal Badu**
- GitHub: [@Himal-Badu](https://github.com/Himal-Badu)
- Twitter: [@himal_badu666](https://twitter.com/himal_badu666)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project useful, give it a star!**
