import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# LOAD DATASET
# -----------------------------

data = pd.read_csv("data/Placement_Data_Full_Class.csv")

print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())


# -----------------------------
# DATA VISUALIZATION
# -----------------------------

sns.countplot(x="status", data=data)
plt.title("Placed vs Not Placed Students")
plt.show()


# -----------------------------
# DATA PREPROCESSING
# -----------------------------

# Drop columns that are not useful
data = data.drop(["sl_no", "salary"], axis=1)

# Convert categorical columns into numbers
data = pd.get_dummies(data, drop_first=True)


# -----------------------------
# SPLIT DATA
# -----------------------------

X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# TRAIN MODEL
# -----------------------------

model = RandomForestClassifier()
model.fit(X_train, y_train)


# -----------------------------
# MODEL EVALUATION
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)


# -----------------------------
# SAVE MODEL
# -----------------------------

os.makedirs("model", exist_ok=True)

with open("model/placement_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")