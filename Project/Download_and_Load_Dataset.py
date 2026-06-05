import pandas as pd

df = pd.read_csv("Fertilizer Prediction.csv")

print("="*60)
print("FERTILIZER PREDICTION DATASET")
print("="*60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 10 Rows:")
print(df.head(10))

print("\nLast 10 Rows:")
print(df.tail(10))

print("\nDataset Information:")
print(df.info())

print("\nMemory Usage:")
print(df.memory_usage(deep=True))