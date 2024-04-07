import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constants
K_VALUE = 8


def find_best_params(input_file, pipe, X, y):
    # Model training with GridSearchCV for hyperparameter tuning
    print("Finding best params for:", input_file)
    param_grid = {
        'model__n_estimators': [50, 100, 150],
        'model__max_depth': [3, 4, 5],
        'model__learning_rate': [0.05, 0.1, 0.15],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
        'model__gamma': [0, 0.1, 0.2]
    }

    # Update the CV
    CV = GridSearchCV(pipe, param_grid, cv=5, n_jobs=8, verbose=2)
    CV.fit(X, y)

    file_name = input_file.removesuffix(".xlsx")

    # Print best params to file
    f = open(file_name + "_best_params.txt", "w")
    f.write(str(CV.best_params_))
    f.close()

    return CV.best_params_


# Prompt user for the data file
input_file = ""
while "xlsx" not in input_file:
    input_file = input("Insert the data file: ")

do_cv = False
best_params = None

try:
    # Check for best params file
    f = open(input_file.removesuffix(".xlsx") + "_best_params.txt")
except FileNotFoundError:
    # No best params, make some
    do_cv = True
else:
    do_cv = False
    best_params = eval(f.read())
    print("Best params already found:", best_params)
    f.close()

xls = pd.ExcelFile(input_file)

# Read training and test data
df = pd.read_excel(xls, sheet_name='train', header=0)
df_test = pd.read_excel(xls, sheet_name='test', header=0)

print("File loaded!")

# Separate numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

# Separate isFraud so it doesn't get processed
numerical_features = numerical_features.drop('isFraud')

categorical_features = df.select_dtypes(include=['object']).columns

# Preprocessing steps
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
feature_selector = SelectKBest(f_classif, k=K_VALUE)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('select', feature_selector, numerical_features)
    ]
)

# Define model with evaluation metric
model = xgb.XGBClassifier(verbosity=1, eval_metric='logloss', tree_method='hist')

# Create preprocessing and modeling pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split data
X = df.drop('isFraud', axis=1)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best model parameters
if do_cv:
    best_params = find_best_params(input_file, pipeline, X_train, y_train)
    print('Best parameters:', best_params)

# Apply the best parameters and make predictions
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Write predictions to the test sheet of the Excel file
df_test['isFraud'] = y_pred
df_test.to_excel(input_file, sheet_name='test', index=False)

# Print model metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: + {precision_score(y_test, y_pred)}")
print(f"Recall: + {recall_score(y_test, y_pred)}")
print(f"F1 Score: + {f1_score(y_test, y_pred)}")
