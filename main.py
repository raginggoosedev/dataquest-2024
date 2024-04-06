import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Prompt user for the data file
input_file = input("Insert the data file: ")

xls = pd.ExcelFile(input_file)

# Comma delimited is default
dataset = pd.read_excel(xls, sheet_name='train', header=0)

#dataset = pd.read_csv(input_file, header=0)

# Separate categorical data from numerical data
categorical_data = dataset.select_dtypes(include=['object'])
numerical_data = dataset.select_dtypes(exclude=['object'])

# Preprocessing
# Create an instance of SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')

# Process data in batches
batch_size = 10000
imputed_batches = []
for i in range(0, len(categorical_data), batch_size):
    batch = categorical_data.iloc[i:i+batch_size]
    imputed_batch = cat_imputer.fit_transform(batch)
    imputed_batches.append(imputed_batch)

# Combine imputed batches
cat_data_imputed = np.vstack(imputed_batches)

# Use one-hot encoding to turn categorical data into usable data
encoder = OneHotEncoder(sparse_output=True)
encoded_data = encoder.fit_transform(categorical_data)

# Create a DataFrame with the encoded data
encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=encoder.get_feature_names_out())

# Combine with numerical data
processed_data = pd.concat([encoded_df, numerical_data], axis=1)

print(processed_data.head())
