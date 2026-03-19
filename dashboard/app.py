import pandas as pd
import pickle

# Load dataset from data folder
data = pd.read_csv("data/Placement_Data_Full_Class.csv")

# Drop unwanted columns
data = data.drop(columns=["sl_no", "salary"])

# Convert categorical to numerical
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]

# Train model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("placement_model.pkl", "wb"))

print("✅ Model trained and saved!")
