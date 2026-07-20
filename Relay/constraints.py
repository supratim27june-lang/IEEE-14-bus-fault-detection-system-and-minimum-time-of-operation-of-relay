"""
constraints.py

Relay coordination constraints and penalties for PSO optimization.

Key corrections vs. the earlier version
----------------------------------------
* The coordination-time-interval (CTI) constraint is now enforced here, and
  each primary/backup pair is evaluated at the SHARED downstream fault
  current it actually sees -- not each relay at its own current. Comparing
  relays at different currents is not a coordination margin and was the
  source of misleading "Satisfied"/"Needs adjustment" results.
* penalty() is now GRADED (proportional to how much each constraint is
  violated) instead of a flat 1e6 cliff. A flat cliff gives PSO no gradient
  toward feasibility; a graded penalty lets the swarm walk into the feasible
  region. CTI dominates lexicographically, then bounds, then a small time
  term, so feasibility is found before time is trimmed.
* Bounds live here only. particle.py imports them, so the two can never
  drift (the old code had TDS_MIN = 0.04 in particle.py vs 0.05 here, which
  silently made floor-hugging particles permanently infeasible).
"""

TDS_MIN = 0.05
TDS_MAX = 1.20
# PICKUP_MIN was 0.50, a leftover from the placeholder reference currents
# (2-9 kA). objective.py now uses real bolted short-circuit duty, and the OC
# (overload) zone current is as low as ~0.245 kA at the last relay -- with
# PICKUP_MIN above that, no pickup below the OC fault current was reachable,
# so the OC/fallback group could never pick up at all. Lowered to sit below
# the smallest real zone current with margin.
PICKUP_MIN = 0.20
PICKUP_MAX = 5.00
COORDINATION_TIME = 0.30
NUM_RELAYS = 5

W_CTI = 1.0e4
W_BOUNDS = 1.0e3
W_TIME = 0.1

from relay_model import Relay
_relay = Relay()


def _iter_relay_settings(position):
    if len(position) != 2 * NUM_RELAYS:
        raise ValueError(f"Expected {2 * NUM_RELAYS} values for {NUM_RELAYS} relays.")
    for relay_idx in range(0, len(position), 2):
        yield position[relay_idx], position[relay_idx + 1]


def cti_shortfall(position, zone_currents):
    """Total CTI shortfall (s) and per-pair margins, each pair at the shared
    downstream current zone_currents[k+1]."""
    total = 0.0
    margins = []
    for k in range(NUM_RELAYS - 1):
        i_down = zone_currents[k + 1]
        tds_b, pu_b = position[2 * k], position[2 * k + 1]
        tds_p, pu_p = position[2 * (k + 1)], position[2 * (k + 1) + 1]
        if pu_b >= i_down or pu_p >= i_down:
            total += 1.0
            margins.append(None)
            continue
        t_backup = _relay.relay_operating_time(i_down, pu_b, tds_b)
        t_primary = _relay.relay_operating_time(i_down, pu_p, tds_p)
        margin = t_backup - t_primary
        margins.append(margin)
        if margin < COORDINATION_TIME:
            total += (COORDINATION_TIME - margin)
    return total, margins


def is_coordinated(position, zone_currents, tol=1e-6):
    total, _ = cti_shortfall(position, zone_currents)
    return total <= tol


def bounds_violation(position):
    v = 0.0
    for tds, pickup in _iter_relay_settings(position):
        v += max(0.0, TDS_MIN - tds) + max(0.0, tds - TDS_MAX)
        v += max(0.0, PICKUP_MIN - pickup) + max(0.0, pickup - PICKUP_MAX)
    return v


def penalty(position, zone_currents):
    """Graded penalty. Zero only when in-bounds AND fully coordinated.
    Signature changed to (position, zone_currents)."""
    cti, _ = cti_shortfall(position, zone_currents)
    bnd = bounds_violation(position)
    return W_CTI * cti + W_BOUNDS * bnd


def print_constraints():
    print("\n========== Relay Constraints ==========")
    print(f"TDS Range            : {TDS_MIN} - {TDS_MAX}")
    print(f"Pickup Current Range : {PICKUP_MIN} - {PICKUP_MAX}")
    print(f"Number of Relays     : {NUM_RELAYS}")
    print("Each pair checked at the shared downstream fault current")
    print(f"Coordination Margin  : {COORDINATION_TIME} s")
    print("=======================================\n")