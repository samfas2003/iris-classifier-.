# src/train.py
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

print("🚀 Starting the Iris Model Training Script...")

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Ensure outputs folder exists and log success
os.makedirs("outputs", exist_ok=True)
print("✅ Model trained successfully! Script completed.")