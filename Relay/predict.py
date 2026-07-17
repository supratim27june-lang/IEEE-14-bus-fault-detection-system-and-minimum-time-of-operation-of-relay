import joblib
import pandas as pd

from features import FEATURES  # same list training used

# -------------------------------------------------
# Load trained Random Forest model
# -------------------------------------------------
model = joblib.load("rf_fault_classifier.pkl")

# -------------------------------------------------
# Guard: fail loudly if the saved model was trained on a
# different feature set/order than features.py declares.
# This turns a silent confidence collapse into an obvious error.
# -------------------------------------------------
if hasattr(model, "feature_names_in_"):
    trained_features = list(model.feature_names_in_)
    if trained_features != FEATURES:
        raise ValueError(
            "Feature mismatch between the saved model and features.py.\n"
            f"  Model was trained on : {trained_features}\n"
            f"  features.py declares : {FEATURES}\n"
            "Re-run RandomForest.py so the two agree."
        )


def predict_fault(
        fault_bus,
        fault_impedance,
        pre_fault_voltage,
        fault_voltage,
        fault_current,   # kept in the signature for caller compatibility
        loading):

    # Build every value we might have available...
    row = {
        "fault_bus": fault_bus,
        "fault_impedance_ohm": fault_impedance,
        "pre_fault_voltage_pu": pre_fault_voltage,
        "fault_voltage_pu": fault_voltage,
        "loading_pu": loading,
        "fault_current_ka": fault_current,  # not a model input (see features.py)
    }

    # ...then keep ONLY the trained features, in the exact trained order.
    # reindex drops fault_current_ka and enforces column order, which is
    # the actual fix for the low-confidence problem.
    X = pd.DataFrame([row]).reindex(columns=FEATURES)

    if X.isna().any(axis=None):
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values for required feature(s): {missing}")

    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max()

    return prediction, confidence