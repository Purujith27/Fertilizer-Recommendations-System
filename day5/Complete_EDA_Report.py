import pandas as pd

df = pd.read_csv("Fertilizer Prediction.csv")

print("="*60)
print("EDA REPORT")
print("="*60)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))

for col in df.select_dtypes(include="object"):
    print(f"\n{'='*40}")
    print(f"VALUE COUNTS OF {col}")
    print(f"{'='*40}")
    print(df[col].value_counts())

print("\nGrouped Analysis:")
print(df.groupby("Fertilizer Name").mean(numeric_only=True))

print("\nTop Fertilizers:")
print(df["Fertilizer Name"].value_counts())

# OBSERVATIONS
print("\nOBSERVATIONS")
print("1. Dataset contains different fertilizer classes.")
print("2. Soil type affects fertilizer recommendation.")
print("3. Crop type influences output fertilizer.")
print("4. N, P, K values show strong variation.")
print("5. Dataset appears balanced across classes.")