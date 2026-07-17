import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import FEATURES  # same list inference uses

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv("fault_dataset_60000.csv")

# -------------------------------------------------
# Features and Target
# -------------------------------------------------
# Selecting df[FEATURES] both guarantees the column ORDER and will raise
# a clear KeyError if the dataset is missing an expected feature.
X = df[FEATURES]
y = df["fault_type"]

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# Random Forest
# -------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

rf.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate
# -------------------------------------------------
pred = rf.predict(X_test)

print("\nAccuracy =", accuracy_score(y_test, pred))

print("\nClassification Report\n")
print(classification_report(y_test, pred))

# -------------------------------------------------
# Save Model
# -------------------------------------------------
joblib.dump(rf, "rf_fault_classifier.pkl")

print("\nRandom Forest model saved successfully.")
print("Trained on features:", list(rf.feature_names_in_))