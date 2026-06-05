import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Fertilizer Prediction.csv")

print("Missing Values Before Cleaning:")
print(df.isnull().sum())

for col in df.columns:
    if is_string_dtype(df[col]) or df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    elif is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

encoders = {}

for col in df.select_dtypes(include="object"):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\nEncoded Dataset Preview:")
print(df.head())

print("\nDataset Shape After Cleaning:")
print(df.shape)