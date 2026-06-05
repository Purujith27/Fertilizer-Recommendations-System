import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = os.path.join(os.path.dirname(__file__), "visuals")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "Fertilizer Prediction.csv"))

# 1
plt.figure(figsize=(8,5))
df["Fertilizer Name"].value_counts().plot(kind="bar")
plt.title("Fertilizer Distribution")
plt.savefig(os.path.join(output_dir, "graph1.png"))
plt.clf()

# 2
plt.figure(figsize=(8,5))
sns.countplot(x="Soil Type", data=df)
plt.title("Soil Type Distribution")
plt.savefig(os.path.join(output_dir, "graph2.png"))
plt.clf()

# 3
plt.figure(figsize=(8,5))
sns.countplot(x="Crop Type", data=df)
plt.xticks(rotation=45)
plt.title("Crop Type Distribution")
plt.savefig(os.path.join(output_dir, "graph3.png"))
plt.clf()

# 4
plt.figure(figsize=(8,5))
sns.boxplot(x="Fertilizer Name", y="Nitrogen", data=df)
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "graph4.png"))
plt.clf()

# 5
plt.figure(figsize=(10,8))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True
)
plt.savefig(os.path.join(output_dir, "graph5.png"))
plt.clf()

print(f"Saved visuals to: {output_dir}")
