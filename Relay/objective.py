"""
objective.py

Coordination objective + fault-type-aware setting-group logic.

Architecture (matches the paper)
--------------------------------
1. Predicted fault TYPE selects a setting-group scenario: a per-relay (zone)
   fault-current vector plus the minimum current that group's relays must
   still detect. Different types -> different levels -> different settings.
2. Model CONFIDENCE is a trust gate. >= CONFIDENCE_THRESHOLD -> use the
   predicted type's group; below it -> a conservative FALLBACK group that
   must detect the weakest fault of ANY type (uncertainty -> MORE conservative).
3. The objective is pure protection physics (no classifier call inside):
   small total-time term + graded penalties for CTI, pickup-window, and
   bound violations. CTI is checked per primary/backup pair at the SHARED
   downstream current (the corrected definition, lives in constraints.py).

Replace-with-real-data hooks: TYPE_REF_CURRENT / TYPE_MIN_CURRENT (clean
coordination reference currents) and the zone-current attenuation model. Swap
for real short-circuit studies per relay location (max/min gen) via
fault.extract_relay_currents for the journal version; interfaces don't change.
"""

from relay_model import Relay
from constraints import (
    NUM_RELAYS, cti_shortfall, bounds_violation,
    W_CTI, W_BOUNDS, W_TIME,
)

relay = Relay()

CONFIDENCE_THRESHOLD = 0.70
ZONE_ATTENUATION = 0.85

# -------------------------------------------------------------------------
# COORDINATION REFERENCE CURRENTS (kA)
#
# IMPORTANT: these are the *coordination* design currents, and they are kept
# CLEAN and physically ordered on purpose. They are NOT taken from the
# classifier dataset (fault_dataset_real.csv): that column is deliberately
# noise-dominated for realistic classification (means ~60-70 kA, outliers to
# thousands of kA, and fault types nearly indistinguishable in magnitude), so
# it is the wrong source for coordination inputs -- using it would collapse the
# per-type interlink and drive relays to their pickup floor.
#
# The physically-correct ordering is LLL (three-phase, highest) > SLG ~ LL >
# OC (overload, lowest). These representative values preserve that ordering so
# different predicted types produce genuinely different coordinated settings.
#
# REPLACE with real per-relay short-circuit currents (fault.extract_relay_currents,
# min-generation case) for the journal version -- that makes the 5-relay
# currents physical rather than a single attenuated base. Interfaces do not change.
# -------------------------------------------------------------------------
TYPE_REF_CURRENT = {   # characteristic fault current that drives operating times
    "LLL": 9.0,
    "SLG": 7.5,
    "LL": 7.0,
    "OC": 2.0,
}
FALLBACK_REF_CURRENT = min(TYPE_REF_CURRENT.values())

TYPE_MIN_CURRENT = {   # minimum current the group's relays must still detect
    "LLL": 7.5,
    "SLG": 6.0,
    "LL": 5.5,
    "OC": 1.5,
}
FALLBACK_MIN_CURRENT = min(TYPE_MIN_CURRENT.values())


def zone_currents_for_scenario(fault_type, confidence):
    """Per-relay fault-current profile (kA) selected by the ML output.

    Confidence >= threshold -> the predicted type's characteristic profile.
    Below -> a conservative worst-case (lowest-current) profile, which is the
    hardest to coordinate and detect, i.e. uncertainty -> MORE conservative.
    """
    if confidence >= CONFIDENCE_THRESHOLD and fault_type in TYPE_REF_CURRENT:
        base = TYPE_REF_CURRENT[fault_type]
    else:
        base = FALLBACK_REF_CURRENT
    return [base * (ZONE_ATTENUATION ** k) for k in range(NUM_RELAYS)]


PICKUP_DETECT_MARGIN = 2.0 / 3.0
PICKUP_LOAD_MARGIN = 1.25
W_PICKUP = 1.0e2


def select_scenario(fault_type, confidence):
    """Return (group_label, min_detect_current_kA) after the trust gate."""
    if confidence >= CONFIDENCE_THRESHOLD and fault_type in TYPE_MIN_CURRENT:
        return fault_type, TYPE_MIN_CURRENT[fault_type]
    return "FALLBACK", FALLBACK_MIN_CURRENT


def build_zone_currents(fault_current):
    """Per-relay fault current down the radial chain (kA)."""
    return [fault_current * (ZONE_ATTENUATION ** k) for k in range(NUM_RELAYS)]


def _pickup_window_penalty(position, zone_currents, min_detect, loading):
    pen = 0.0
    ceiling = PICKUP_DETECT_MARGIN * min_detect
    for k in range(NUM_RELAYS):
        pu = position[2 * k + 1]
        if pu > ceiling:
            pen += (pu - ceiling)
        load_floor = PICKUP_LOAD_MARGIN * loading * (ZONE_ATTENUATION ** k)
        if pu < load_floor:
            pen += (load_floor - pu)
        if pu >= zone_currents[k]:
            pen += (pu - zone_currents[k] + 1.0)
    return pen


def coordination_objective(position, zone_currents, min_detect, loading):
    total_time = 0.0
    for k in range(NUM_RELAYS):
        tds, pu = position[2 * k], position[2 * k + 1]
        i = zone_currents[k]
        total_time += 50.0 if pu >= i else relay.relay_operating_time(i, pu, tds)

    cti, _ = cti_shortfall(position, zone_currents)
    bnd = bounds_violation(position)
    pick = _pickup_window_penalty(position, zone_currents, min_detect, loading)
    return W_CTI * cti + W_BOUNDS * bnd + W_PICKUP * pick + W_TIME * total_time


def adaptive_objective(
    position, fault_current, loading, fault_type,
    fault_bus, fault_impedance, pre_fault_voltage, fault_voltage,
    confidence=1.0,
):
    """PSO entry point. Caller classifies ONCE and passes (fault_type,
    confidence) in; the objective never calls the model.

    INTERLINK: the predicted type (gated by confidence) selects BOTH the
    per-relay fault-current profile (zone_currents_for_scenario) AND the
    minimum-detectable current (select_scenario). The current profile drives
    every operating time, so the ML decision genuinely reshapes the
    optimization -- different types yield different coordinated settings.
    """
    _group, min_detect = select_scenario(fault_type, confidence)
    zone_currents = zone_currents_for_scenario(fault_type, confidence)
    return coordination_objective(position, zone_currents, min_detect, loading)