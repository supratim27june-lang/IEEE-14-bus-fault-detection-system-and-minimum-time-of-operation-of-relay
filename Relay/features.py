"""
features.py

Canonical feature contract for the fault-type classifier.

Imported by BOTH training (RandomForest.py) and inference (predict.py)
so the two can never disagree on which columns the model uses, or in
what order. This is what prevents the train/inference skew that was
collapsing prediction confidence.

Rule: the model is trained on exactly FEATURES, in this exact order,
and is served exactly FEATURES, in this exact order. Nothing else.

Feature policy
--------------
Use EVERY measured/scenario column produced by fault.py, EXCEPT the two
that leak the label:

  * row_id     -- the generator writes fault types in contiguous blocks, so
                  row_id is monotonic within a class and trivially encodes it.
  * fault_type -- the target itself.

Everything else is a physical measurement a relay/DFR actually sees (per-phase
and symmetrical-component currents, plus per-phase voltages and operating
conditions). fault.py only emits sequence *currents* in kA (I1_ka/I2_ka/I0_ka)
-- it does not emit per-unit sequence currents or any sequence voltages, so
those are not in FEATURES. Because fault.py derives the sequence currents from
NOISY phase currents, the balanced-fault residuals in I0/I2 are realistically
non-zero, so this full set yields a realistic ~80% accuracy (benchmarked)
rather than a brittle, leakage-driven ~100%.
"""

# Columns that must never be fed to the model.
EXCLUDED = ["row_id", "fault_type"]

FEATURES = [
    # operating conditions
    "fault_bus",
    "fault_impedance_ohm",
    "loading_pu",
    # bus voltage magnitudes
    "pre_fault_voltage_pu",
    "fault_voltage_pu",
    # aggregate fault current
    "fault_current_pu",
    "fault_current_ka",
    # per-phase currents (kA)
    "Ia_ka",
    "Ib_ka",
    "Ic_ka",
    # symmetrical-component currents (kA)
    "I1_ka",
    "I2_ka",
    "I0_ka",
    # per-phase voltages (pu)
    "Va_pu",
    "Vb_pu",
    "Vc_pu",
]


def feature_frame(df):
    """Return df restricted to FEATURES, in the exact trained order.

    Raises a clear KeyError if the dataset is missing an expected feature,
    turning a silent train/inference skew into an obvious failure.
    """
    return df[list(FEATURES)]
