#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from flask import Flask, request, render_template, flash, redirect, url_for
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import joblib
import os
import secrets
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'Uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# In[ ]:


# Load the dataset
data = pd.read_csv('C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\dataset.csv')


# In[ ]:


# Compute unique values for categorical columns
categorical_cols = ['ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type',
                    'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']
unique_values = {col: data[col].dropna().unique().tolist() for col in categorical_cols}


# In[ ]:


# Load the trained model and preprocessing objects
model = joblib.load('C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\svm_model.pkl')
scaler = joblib.load('C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\scaler.pkl')
pca = joblib.load('C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\pca.pkl')
imputer_numeric = joblib.load('C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\imputer_numeric.pkl')
imputer_categorical = joblib.load('C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\imputer_categorical.pkl')
label_encoders = joblib.load('C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\label_encoders.pkl')


# In[ ]:


# Define expected columns (excluding target and IDs)
expected_columns = [
    'age', 'bmi', 'elective_surgery', 'ethnicity', 'gender', 'height',
    'icu_admit_source', 'icu_stay_type', 'icu_type', 'pre_icu_los_days',
    'weight', 'apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative',
    'arf_apache', 'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache',
    'gcs_verbal_apache', 'heart_rate_apache', 'intubated_apache', 'map_apache',
    'resprate_apache', 'temp_apache', 'ventilated_apache', 'd1_diasbp_max',
    'd1_diasbp_min', 'd1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min',
    'd1_heartrate_max', 'd1_heartrate_min', 'd1_mbp_max', 'd1_mbp_min',
    'd1_mbp_noninvasive_max', 'd1_mbp_noninvasive_min', 'd1_resprate_max',
    'd1_resprate_min', 'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_max',
    'd1_sysbp_min', 'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min',
    'd1_temp_max', 'd1_temp_min', 'h1_diasbp_max', 'h1_diasbp_min',
    'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 'h1_heartrate_max',
    'h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max',
    'h1_mbp_noninvasive_min', 'h1_resprate_max', 'h1_resprate_min', 'h1_spo2_max',
    'h1_spo2_min', 'h1_sysbp_max', 'h1_sysbp_min', 'h1_sysbp_noninvasive_max',
    'h1_sysbp_noninvasive_min', 'd1_glucose_max', 'd1_glucose_min',
    'd1_potassium_max', 'd1_potassium_min', 'apache_4a_hospital_death_prob',
    'apache_4a_icu_death_prob', 'aids', 'cirrhosis', 'diabetes_mellitus',
    'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma',
    'solid_tumor_with_metastasis', 'apache_3j_bodysystem', 'apache_2_bodysystem'
]


# In[ ]:


# Function to preprocess input data
def preprocess_data(df):
    print("Uploaded dataset columns:", df.columns.tolist())
    print("Shape before preprocessing:", df.shape)

    # Explicitly drop problematic columns if present
    columns_to_drop = ['encounter_id', 'hospital_id', 'patient_id', 'hospital_death', 'icu_id', 'Unnamed: 83']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    print("Columns after dropping IDs and target:", df.columns.tolist())
    print("Shape after dropping:", df.shape)

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_columns]
    print("Final columns before scaling:", df.columns.tolist())
    print("Shape after ensuring columns:", df.shape)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = imputer_numeric.transform(df[numeric_cols])
    df[categorical_cols] = imputer_categorical.transform(df[categorical_cols])
    print("Shape after imputation:", df.shape)

    # Encode categorical variables
    for col in categorical_cols:
        le = label_encoders[col]
        unseen_values = set(df[col].astype(str)) - set(le.classes_)
        if unseen_values:
            print(f"Warning: Unseen values in {col}: {unseen_values}. Mapping to {le.classes_[0]}")
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col].astype(str))
    print("Shape after encoding:", df.shape)

    # Standardize features
    try:
        X_scaled = scaler.transform(df)
        print("Shape after scaling:", X_scaled.shape)
    except Exception as e:
        print("Error during scaling:", str(e))
        raise e

    # Apply PCA
    try:
        X_pca = pca.transform(X_scaled)
        print("Shape after PCA:", X_pca.shape)
        print("Number of PCA components:", X_pca.shape[1])
    except Exception as e:
        print("Error during PCA:", str(e))
        raise e

    return X_pca


# In[ ]:


resulted_data = preprocess_data(data)


# In[ ]:


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# In[ ]:


# File upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))

    if file and file.filename.endswith('.csv'):
        try:
            # Save uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Read and preprocess data
            df = pd.read_csv(file_path)
            X_processed = preprocess_data(df)

            # Make predictions
            predictions = model.predict(X_processed)

            # Debug: Print prediction distribution
            pred_counts = pd.Series(predictions).value_counts(normalize=True)
            print("Prediction distribution:", pred_counts.to_dict())

            # Prepare results
            results = pd.DataFrame({
                'Record': range(1, len(predictions) + 1),
                'Prediction': ['Died' if pred == 1 else 'Survived' for pred in predictions]
            })

            return render_template('results.html', tables=[results.to_html(classes='data', index=False)], titles=results.columns.values)

        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('home'))
    else:
        flash('Please upload a CSV file')
        return redirect(url_for('home'))


# In[ ]:


# Manual input route
@app.route('/manual', methods=['GET', 'POST'])
def manual_input():
    if request.method == 'POST':
        try:
            # Collect form data
            input_data = {}
            for col in expected_columns:
                value = request.form.get(col)
                input_data[col] = [float(value) if col not in categorical_cols else value]
            
            # Create DataFrame
            df = pd.DataFrame(input_data)
            print("Manual input DataFrame shape:", df.shape)

            # Preprocess data
            X_processed = preprocess_data(df)
            print("Processed input shape:", X_processed.shape)

            # Make prediction
            prediction = model.predict(X_processed)[0]
            result = 'Died' if prediction == 1 else 'Survived'

            return render_template('result_manual.html', prediction=result)

        except Exception as e:
            flash(f'Error processing input: {str(e)}')
            return redirect(url_for('manual_input'))

    return render_template('manual_input.html', 
                           columns=expected_columns, 
                           categorical_cols=categorical_cols, 
                           unique_values=unique_values)


# In[ ]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




