"""
constraints.py

Relay coordination constraints for PSO optimization.
"""

# -------------------------------
# Relay Parameter Limits
# -------------------------------

TDS_MIN = 0.05
TDS_MAX = 1.20

PICKUP_MIN = 0.50
PICKUP_MAX = 5.00

# Relay coordination margin (seconds)
COORDINATION_TIME = 0.30


def is_feasible(position, fault_current):
    """
    Check whether a relay setting is feasible.

    Parameters
    ----------
    position : list or tuple
        [TDS, Pickup_Current]

    fault_current : float
        Fault current (pu)

    Returns
    -------
    bool
        True if all constraints are satisfied.
    """

    tds = position[0]
    pickup = position[1]

    # --------------------------
    # Constraint 1 : TDS Range
    # --------------------------
    if tds < TDS_MIN or tds > TDS_MAX:
        return False

    # --------------------------
    # Constraint 2 : Pickup Range
    # --------------------------
    if pickup < PICKUP_MIN or pickup > PICKUP_MAX:
        return False

    # --------------------------
    # Constraint 3 : Relay should
    # actually detect the fault
    # --------------------------
    if fault_current <= pickup:
        return False

    return True


def penalty(position, fault_current):
    """
    Penalty function for infeasible solutions.

    Returns
    -------
    float
        0 if feasible
        Large penalty otherwise
    """

    if is_feasible(position, fault_current):
        return 0

    return 1e6


def print_constraints():
    """
    Print all optimization constraints.
    """

    print("\n========== Relay Constraints ==========")

    print(f"TDS Range            : {TDS_MIN} - {TDS_MAX}")

    print(f"Pickup Current Range : {PICKUP_MIN} - {PICKUP_MAX}")

    print("Fault Current > Pickup Current")

    print(f"Coordination Margin  : {COORDINATION_TIME} s")

    print("=======================================\n")