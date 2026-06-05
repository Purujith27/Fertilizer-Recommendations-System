import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Resolve paths relative to this script so it works from any CWD
base = Path(__file__).resolve().parent
df = pd.read_csv(base / "Fertilizer Prediction.csv")

for col in df.select_dtypes(include="object"):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

with open(base / "model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully")