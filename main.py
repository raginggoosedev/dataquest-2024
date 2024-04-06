import numpy as np
import pandas as pd

input_file = input("Insert the data file: ")

# Comma delimited is default
df = pd.read_csv(input_file, header=0)

print(df.head())