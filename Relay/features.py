"""
features.py

Canonical feature contract for the fault-type classifier.

Imported by BOTH training (RandomForest.py) and inference (predict.py)
so the two can never disagree on which columns the model uses, or in
what order. This is what prevents the train/inference skew that was
collapsing prediction confidence to ~0.47.

Rule: the model is trained on exactly FEATURES, in this exact order,
and is served exactly FEATURES, in this exact order. Nothing else.
"""

FEATURES = [
    "fault_bus",
    "fault_impedance_ohm",
    "pre_fault_voltage_pu",
    "fault_voltage_pu",
    "loading_pu",
]

# If you later decide fault current SHOULD be a model input, add it here
# (e.g. "fault_current_ka") and simply re-run RandomForest.py. Both training
# and inference will pick it up automatically and stay consistent.