import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

# Prompt user for the data file
input_file = ""
while "xlsx" not in input_file:
    input_file = input("Insert the data file: ")

xls = pd.ExcelFile(input_file)

# Read training and test data
df = pd.read_excel(xls, sheet_name='train', header=0)
df_test = pd.read_excel(xls, sheet_name='test', header=0)

print("Files loaded!")

# Separate numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

# Separate isFraud so it doesn't get processed
numerical_features = numerical_features.drop('isFraud')

categorical_features = df.select_dtypes(include=['object']).columns

# Preprocessing steps
k = 8

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
feature_selector = SelectKBest(f_classif, k=k)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('select', feature_selector, numerical_features)
    ]
)

# Define model with evaluation metric and early stopping
model = xgb.XGBClassifier(verbosity=1, eval_metric='logloss', tree_method='hist')

# Create preprocessing and modeling pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split data
X = df.drop('isFraud', axis=1)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with GridSearchCV for hyperparameter tuning
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 4],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0],
    'model__gamma': [0, 0.1]
}

# Update the CV without specifying early stopping
CV = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=2, verbose=1)
CV.fit(X_train, y_train)

# Best model parameters
print('Best parameters:', CV.best_params_)
