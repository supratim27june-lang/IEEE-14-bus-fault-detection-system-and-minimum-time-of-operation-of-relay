"""
Adaptive objective function for ML-assisted relay optimization.
"""
from predict import predict_fault
from relay_model import Relay
relay=Relay()


def adaptive_objective(position,
                       fault_current,
                       loading,
                       fault_type,
                       fault_bus,
                       fault_impedance,
                       pre_fault_voltage,
                       fault_voltage):
    fault_type, confidence = predict_fault(
        fault_bus,
        fault_impedance,
        pre_fault_voltage,
        fault_voltage,
        fault_current,
        loading
)

    tds = position[0]
    pickup = position[1]

    operating_time = relay.relay_operating_time(
        fault_current,
        pickup,
        tds
    )

    penalty = 0

    # -------------------------
    # Loading penalty
    # -------------------------

    if loading > 0.90:

        if pickup < 2.5:

            penalty += 2.0

    elif loading > 0.75:

        if pickup < 2.0:

            penalty += 1.0

    # -------------------------
    # Fault type penalty
    # -------------------------

    if fault_type in ["LLL", "LLLG"]:

        if operating_time > 0.25:

            penalty += 3

    elif fault_type == "SLG":

        if operating_time > 0.40:

            penalty += 2

    # -------------------------
    # Bus penalty
    # -------------------------

    if fault_bus >= 10:

        if tds > 0.5:

            penalty += 1

    return operating_time + penalty