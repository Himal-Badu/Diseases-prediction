# Heart Disease Prediction Toolkit

This project implements a machine learning-based heart disease prediction toolkit that can predict thalassemia types using various classification algorithms.

## Dataset Used

**Heart Disease Dataset**: Preprocessed version with one-hot encoding
- Contains clinical attributes related to heart health
- Target variable: thalassemia type (fixed defect, normal, reversable defect)
- Features include age, blood pressure, cholesterol levels, and other clinical measurements
- Dataset shape: 5 rows, 17 columns
- Class distribution: {1: np.int64(3), 0: np.int64(1), 2: np.int64(1)}

## Models Implemented

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Performance Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score (where applicable)

## Results Summary

### Heart Disease Prediction
- Logistic Regression: Accuracy = 0.0000, F1-Score = 0.0000
- Decision Tree: Accuracy = 0.0000, F1-Score = 0.0000
- Random Forest: Accuracy = 1.0000, F1-Score = 1.0000

Best performing model: Random Forest (Accuracy: 1.0000)

## How to Use

1. Clone the repository
2. Install required packages: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Run the Jupyter notebook or Python script to train models
4. Use the prediction function to make predictions on new data

### Making Predictions

```python
import joblib
import numpy as np
import json

# Load the saved model, scaler, and feature names
model = joblib.load('heart_random_forest_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

# Load feature names
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Prepare new data (example - replace with your actual values)
new_data = np.array([[
    58,      # age
    130,     # trestbps
    220,     # chol
    1,       # fbs
    0,       # restecg (encoded)
    150,     # thalch
    0,       # exang
    1.4,     # oldpeak
    0,       # slope (encoded)
    0,       # ca
    0,       # thal (encoded)
    0,       # sex_Female
    1,       # sex_Male
    0,       # cp_asymptomatic
    0,       # cp_atypical angina
    0,       # cp_non-anginal
    1        # cp_typical angina
]])

# Make prediction (you'll need to define the predict_heart_disease function)
prediction, prediction_label, probability = predict_heart_disease(
    model, scaler, new_data, feature_names
)
print(f"Prediction: {prediction_label}")
print(f"Probability: {probability}")
```

## File Structure

- `heart_disease_prediction.ipynb`: Main Jupyter notebook with complete implementation
- `heart_*_model.pkl`: Trained models for heart disease prediction
- `heart_scaler.pkl`: Scaler used for data preprocessing
- `feature_names.json`: List of feature names in the correct order
- `heart_results.json`: Evaluation results for all models
- `summary_report.json`: Summary of the dataset and best model

## Author

Himal Badu

## License

This project is licensed under the MIT License.