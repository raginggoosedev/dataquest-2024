import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load data
xls = pd.ExcelFile('data.xlsx')  # Update this path to your actual file path
train_data = pd.read_excel(xls, sheet_name='train')

# Preprocess data
# One-hot encode categorical variables
categorical_features = ['business', 'category', 'gender', 'state', 'job']
one_hot_encoder = ColumnTransformer(transformers=[('cat', OneHotEncoder(sparse=False), categorical_features)], remainder='passthrough')
train_encoded = one_hot_encoder.fit_transform(train_data)

# Feature Engineering
train_data['age'] = (np.datetime64('2024-01-01') - train_data['dateOfBirth']).astype('timedelta64[Y]')
train_data['dayOfWeek'] = train_data['transDate'].dt.dayofweek
train_data['hourOfDay'] = train_data['transDate'].dt.hour

# Normalize 'amount' feature
scaler = StandardScaler()
train_data['amount_scaled'] = scaler.fit_transform(train_data[['amount']])

# Split data into features and target
features = ['age', 'dayOfWeek', 'hourOfDay', 'amount_scaled']  # Include any other features as needed
X = train_data[features]
y = train_data['isFraud']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier with balanced class weights
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Set up GridSearchCV to find the best parameters for the Random Forest
param_grid = {
    'n_estimators': [300],
    'max_features': ['auto'],
    'max_depth': [20],
    'min_samples_split': [5],
    'min_samples_leaf': [1]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_rf_model = grid_search.best_estimator_

# Predict on the validation set using the best Random Forest model
y_pred_rf = best_rf_model.predict(X_val)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_val, y_pred_rf)
rf_precision = precision_score(y_val, y_pred_rf)
rf_recall = recall_score(y_val, y_pred_rf)
rf_f1 = f1_score(y_val, y_pred_rf)

print(f"Random Forest Best Parameters: {grid_search.best_params_}")
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1 Score: {rf_f1}")
