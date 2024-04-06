import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sys import exit

# Prompt user for the data file
input_file = input("Insert the data file: ")
if ('.' not in input_file): input_file = input_file + ".xlsx"
xls = pd.ExcelFile(input_file)

# Comma delimited is default
dataset = pd.read_excel(xls, sheet_name='train', header=0)
dataset['transDate'] = dataset['transDate'].astype('int64')
dataset['dateOfBirth'] = dataset['dateOfBirth'].astype('int64')


##### PREPROCESSING #####

# Categorical

categorical_data = dataset.select_dtypes(include=['object']).columns

cat_imputer = SimpleImputer(strategy='most_frequent')
cat_onehot = OneHotEncoder(handle_unknown='ignore')
cat_transformer = Pipeline(steps=[('impute', cat_imputer), ('onehot', cat_onehot)])

preprocessor_for_cat_columns = ColumnTransformer(transformers=[('cat', cat_transformer, categorical_data)], remainder="passthrough")

# All columns

numerical_data = dataset.select_dtypes(include=[np.cfloat, np.int64]).columns

num_scaler = StandardScaler()
num_transformer = Pipeline(steps=[('scale', num_scaler)])

preprocessor_for_all_columns = ColumnTransformer(transformers=[('cat', cat_transformer, categorical_data),
                                                                ('num', num_transformer, numerical_data)], remainder="passthrough")


##### MODEL #####

model_name = "Random Forest Classifer"

random_forest_classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

rfc_model = Pipeline(steps=[('preprocessorAll', preprocessor_for_all_columns), ('classifier', random_forest_classifier)])


print(dataset.columns)

from sklearn.model_selection import train_test_split
#X = dataset.drop('isFraud', axis=1)
X = dataset
y = dataset['isFraud']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc_model.fit(X_train, y_train)

y_pred_rfc = rfc_model.predict(X_test)

two_d_compare(y_test, y_pred_rfc, model_name)