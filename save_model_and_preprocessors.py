#!/usr/bin/env python
# coding: utf-8

# ## Importing Required Libraries

# In[ ]:


#Data Manipulation and Preprocessing
import pandas as pd
import numpy as np
#Data Visualiztion
import matplotlib.pyplot as plt
import seaborn as sns
#Building the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
#Save Model Results
import joblib
#Warnings Handler
import warnings
warnings.filterwarnings('ignore')
# Setting random seed for reproducibility
np.random.seed(42)


# In[ ]:


# Load the dataset
data = pd.read_csv('C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\dataset.csv')


# ## Exploratory Data Analysis (EDA)

# In[ ]:


print("\nDataset Shape:", data.shape)
print()
print("\nFirst 5 Rows:")
print(data.head())
print()
print("\nData Info:")
print(data.info())
print()
print("\nSummary Statistics:")
print(data.describe())


# In[ ]:


# Distribution of Hospital Death
plt.figure(figsize=(10, 6))
sns.countplot(x='hospital_death', data=data)
plt.title('Distribution of Hospital Death')
plt.savefig('hospital_death_distribution.png')
plt.close()


# In[ ]:


# Correlation matrix for numeric features
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12, 8))
sns.heatmap(data[numeric_cols].corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()


# In[ ]:


# Distribution of age
plt.figure(figsize=(10, 6))
sns.histplot(data['age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.close()


# In[ ]:


# Boxplot of BMI by hospital death
plt.figure(figsize=(10, 6))
sns.boxplot(x='hospital_death', y='bmi', data=data)
plt.title('BMI vs Hospital Death')
plt.savefig('bmi_vs_hospital_death.png')
plt.close()


# ## Data Cleaning Phase

# In[ ]:


# Drop unnecessary column
data = data.drop(columns=['Unnamed: 83','encounter_id', 'patient_id', 'hospital_id','icu_id'])


# In[ ]:


missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])


# In[ ]:


X = data.drop(columns = ['hospital_death'])
y = data['hospital_death']


# In[ ]:


# Handle missing values
# Numeric columns: Impute with median
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
imputer_numeric = SimpleImputer(strategy='median')
X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])


# In[ ]:


# Categorical columns: Impute with mode
categorical_cols = X.select_dtypes(include=['object']).columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_categorical.fit_transform(X[categorical_cols])


# In[ ]:


# Verify no missing values remain
print("\nMissing Values After Imputation:")
print(X.isnull().sum().sum())


# In[ ]:


# Handle outliers 
def remove_outliers(df, target, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    return df[mask], target[mask]


# In[ ]:


# Apply outlier removal only to numeric columns with more than 2 unique values
numeric_features = X.select_dtypes(include='number')
numeric_features = numeric_features.loc[:, numeric_features.nunique() > 2]
for col in numeric_features:
    X, y = remove_outliers(X, y, col)


# In[ ]:


print("\nShape After Outlier Removal For Features Data:", X.shape)
print("\nShape After Outlier Removal For Target Data:", y.shape)


# ## Building Model Phase

# In[ ]:


# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le


# In[ ]:


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


# Apply PrincipalComponentAnalysis(PCA)
pca = PCA(n_components=0.80)  
X_pca = pca.fit_transform(X_scaled)
print("\nNumber of PCA Components:", X_pca.shape[1])
print("Explained Variance Ratio:", sum(pca.explained_variance_ratio_))


# In[ ]:


# Visualize PCA explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.savefig('pca_explained_variance.png')
plt.close()


# In[ ]:


#OverSampling the target varibale
over_sampling = RandomOverSampler(random_state=42)
X_pca_resampled, y_resampled = over_sampling.fit_resample(X_pca, y)
y_resampled.value_counts()


# In[ ]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_resampled, y_resampled, test_size=0.2, random_state=42)

# Define SVM model and parameter grid
svm = SVC(random_state=42)
param_grid = {
    'C': [1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale']
}

# Perform grid search
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[ ]:


# Best model
best_svm = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)


# In[ ]:


# Evaluate on test set
y_pred = best_svm.predict(X_test)
print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()


# ## Load Model, Scaling, PCA, Imputation, and Enncodinng Results and Save Them In a Directory 

# In[ ]:


# Save the model and preprocessing objects
joblib.dump(grid_search.best_estimator_, 'C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\svm_model.pkl')
joblib.dump(scaler, 'C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\scaler.pkl')
joblib.dump(pca, 'C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\pca.pkl')
joblib.dump(imputer_numeric, 'C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\imputer_numeric.pkl')
joblib.dump(imputer_categorical, 'C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\imputer_categorical.pkl')
joblib.dump(label_encoders, 'C:\\Users\\lenovo\\OneDrive\\Desktop\\Data Computauion Project\\label_encoders.pkl')
print("Model and preprocessing objects saved successfully.")

