import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

# Numerical

numerical_data = dataset.select_dtypes(exclude=['object']).columns

num_scaler = StandardScaler()
num_transformer = Pipeline(steps=[('scale', num_scaler)])

preprocessor_for_num_columns = ColumnTransformer(transformers=[('num', num_transformer, numerical_data)], remainder="passthrough")


df_churn_pd_temp1 = preprocessor_for_cat_columns.fit_transform(dataset)
df_churn_pd_temp2 = preprocessor_for_num_columns.fit_transform(dataset)

