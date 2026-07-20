"""
Fault-type classifier inference.

FEATURE CONTRACT
----------------
The model consumes the full physical feature vector defined in features.py
(per-phase and symmetrical-component currents/voltages plus operating
conditions) -- everything the dataset carries except the two leaky columns
row_id and fault_type.

Because there are now many features, prediction takes a single MAPPING
(a dict or a pandas Series/row) keyed by feature name, rather than a long
positional argument list. A dataset row can be passed straight through:

    from predict import predict_fault
    pred, conf = predict_fault(scenario_row)     # scenario_row is a pd.Series

UNIT CONTRACT
-------------
Currents are in the same units the model was trained on: *_ka in kA, *_pu in
per-unit. Do NOT rescale (the old app multiplied kA by 1000 and destroyed the
current features). Pass measured values straight through.
"""

import joblib
import pandas as pd

from features import FEATURES  # single source of truth for column set/order

# -------------------------------------------------
# Load trained Random Forest model
# -------------------------------------------------
model = joblib.load("rf_fault_classifier.pkl")

# -------------------------------------------------
# Guard: fail loudly if the saved model was trained on a different feature
# set/order than features.py declares. Turns a silent accuracy collapse
# into an obvious, actionable error at import time.
# -------------------------------------------------
if hasattr(model, "feature_names_in_"):
    trained_features = list(model.feature_names_in_)
    if trained_features != list(FEATURES):
        raise ValueError(
            "Feature mismatch between the saved model and features.py.\n"
            f"  Model was trained on : {trained_features}\n"
            f"  features.py declares : {list(FEATURES)}\n"
            "Re-run RandomForest.py so the two agree."
        )


def _vectorize(features):
    """Build the 1-row feature frame the model expects from a mapping.

    features : dict-like keyed by feature name (a dict or a pandas Series).
    Enforces the exact trained column set AND order, and fails clearly if any
    required feature is missing or NaN.
    """
    if hasattr(features, "to_dict") and not isinstance(features, dict):
        features = features.to_dict()

    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    X = pd.DataFrame([{f: features[f] for f in FEATURES}])[list(FEATURES)]

    if X.isna().any(axis=None):
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN value(s) for required feature(s): {nan_cols}")
    return X


def predict_fault(features):
    """Return (predicted_fault_type, confidence) for one operating point.

    features   : dict-like keyed by feature name (see features.FEATURES).
    confidence : the max class probability (RF vote proportion) in [0, 1].
    """
    X = _vectorize(features)
    prediction = model.predict(X)[0]
    confidence = float(model.predict_proba(X)[0].max())
    return prediction, confidence


def predict_distribution(features):
    """Return {class_label: probability} for one operating point.

    Useful if you want the per-class probabilities rather than only the top
    class. Kept separate so predict_fault stays cheap.
    """
    X = _vectorize(features)
    probs = model.predict_proba(X)[0]
    return dict(zip(model.classes_, (float(p) for p in probs)))
