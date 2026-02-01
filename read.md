# Parkinson's Disease Detection Using Voice Frequency

## Project Overview
This project detects Parkinson's disease using voice frequency features. 
We used a dataset from UCI containing recordings of patients’ voices. 

## Dataset
- 195 samples
- 22 voice frequency features
- Target column: `status` (1 = Parkinson, 0 = Healthy)

## Methodology
1. Data preprocessing:
   - Dropped `name` column
   - Scaled features using MinMaxScaler
   - Balanced data using SMOTE
2. Train-test split (80%-20%)
3. Machine Learning Model:
   - Random Forest Classifier
   - Accuracy achieved: 100%
4. Saved trained model as `parkinson_model.pkl`

## Libraries Used
- Python 3
- Pandas, Numpy, Seaborn, Matplotlib
- Scikit-learn, imbalanced-learn, XGBoost

## How to Use
```python
import joblib
model = joblib.load("parkinson_model.pkl")
# Use model.predict(new_data) for predictions
