import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

from features import feature_frame  # same contract inference uses

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv("fault_dataset_hard.csv")

# -------------------------------------------------
# Features and Target
# -------------------------------------------------
# feature_frame() selects exactly FEATURES in order and raises a clear KeyError
# if a column is missing. row_id / fault_type are excluded by the contract
# (row_id leaks the label via block ordering; fault_type is the target).
X = feature_frame(df)
y = df["fault_type"]

# -------------------------------------------------
# Train-Test Split  (stratified, matching the benchmark protocol)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42,
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# -------------------------------------------------
# Random Forest
# -------------------------------------------------
# Config matches Scripts/benchmark.py so the served model reproduces the
# benchmarked ~80% accuracy. This is the HONEST ceiling on this dataset: the
# sequence residuals of balanced faults are noisy, not exact zeros, so the
# classes are not trivially separable.
rf = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

rf.fit(X_train_resampled, y_train_resampled)

# -------------------------------------------------
# Evaluate
# -------------------------------------------------
pred = rf.predict(X_test)

print("\nAccuracy =", accuracy_score(y_test, pred))

print("\nClassification Report\n")
print(classification_report(y_test, pred))

print("Top feature importances:")
for name, imp in sorted(zip(rf.feature_names_in_, rf.feature_importances_),
                        key=lambda t: -t[1])[:10]:
    print(f"    {name:22s} {imp:.4f}")

# -------------------------------------------------
# Save Model
# -------------------------------------------------
joblib.dump(rf, "rf_fault_classifier.pkl")

print("\nRandom Forest model saved successfully.")
print("Trained on features:", list(rf.feature_names_in_))
