import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import os

# Load dataset relative to this script's location so the script works
# regardless of the current working directory.
base_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_dir, "Fertilizer Prediction.csv"))

# Encode non-numeric columns. Use direct dtype checks to avoid a
# Pandas deprecation warning from `select_dtypes(include="object")`.
for col in df.select_dtypes(include=["object", "string"]):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature Importance:")
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print(
    importance.sort_values(
        by="Importance",
        ascending=False
    )
)