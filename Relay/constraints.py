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
NUM_RELAYS = 5

# ---------------------------------------------------------------------------
# Per-relay minimum pickup current, derived from each relay's own CT (current
# transformer) rating -- NOT a single global floor.
#
# IEC/ANSI protection practice: a relay's pickup tap should never be set below
# roughly 1.2-1.5x the CT's rated secondary current -- below that, the CT's
# own measurement accuracy/burden behaviour at low primary current becomes
# unreliable and the setting is not physically meaningful. This replaces the
# earlier single scalar PICKUP_MIN (which had to be squeezed down to 0.002 kA
# to let R5's ~0.043 kA OC current pick up at all -- physically meaningless
# for R1's 400/1 A CT, which never legitimately sees primary current that low).
#
# CT_PRIMARY_RATING_KA is each relay's own CT primary rating (secondary is the
# standard 1 A), sized to its position in the network per type_reference_
# currents.py's Table 4 (min-generation).
#
# The binding requirement is NOT just "clears the relay's own OC current": a
# relay also acts as BACKUP for its immediate downstream neighbour, and CTI
# coordination (constraints.cti_shortfall) requires the backup relay to still
# detect the shared current at that downstream zone. Since OC current falls
# monotonically along the chain (R1=1.133 -> R5=0.043 kA, always the weakest
# class at every position), the downstream neighbour's OC current -- not the
# relay's own -- is the smaller, more binding value for every relay except
# R5 (which has no downstream neighbour). An earlier version of this table
# only checked each relay against its own OC current and set R1/R2 too coarse
# (0.540/0.135 kA) to see R2/R3's OC current (0.408/0.120 kA) -- silently
# breaking R1-R2 and R2-R3 coordination under OC/fallback scenarios. Ratios
# below are sized against min(own OC, downstream neighbour's OC) instead, so
# 1.35x the CT secondary rating (the midpoint of the 1.2-1.5x guideline, used
# as the actual floor) clears the correct current with margin:
#   R1: floor 0.2025 kA < R2's OC 0.408 kA   (backup-limited)
#   R2: floor 0.0540 kA < R3's OC 0.120 kA   (backup-limited)
#   R3: floor 0.0270 kA < R4's OC 0.073 kA   (backup-limited)
#   R4: floor 0.0135 kA < R5's OC 0.043 kA   (backup-limited)
#   R5: floor 0.0108 kA < R5's own OC 0.043 kA (no downstream neighbour)
CT_RATED_SECONDARY_A = 1.0                          # A, standard IEC secondary
CT_PRIMARY_RATING_KA = [0.150, 0.040, 0.020, 0.010, 0.008]  # R1..R5
CT_MIN_PICKUP_FACTOR = 1.35                         # midpoint of 1.2-1.5x

PICKUP_MIN = [round(CT_MIN_PICKUP_FACTOR * ct, 5) for ct in CT_PRIMARY_RATING_KA]
PICKUP_MAX = 5.00
COORDINATION_TIME = 0.30

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
    for k, (tds, pickup) in enumerate(_iter_relay_settings(position)):
        v += max(0.0, TDS_MIN - tds) + max(0.0, tds - TDS_MAX)
        v += max(0.0, PICKUP_MIN[k] - pickup) + max(0.0, pickup - PICKUP_MAX)
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
    print("Pickup Current Range (per relay, CT-derived minimum):")
    for k in range(NUM_RELAYS):
        print(f"  R{k+1} (CT {CT_PRIMARY_RATING_KA[k]*1000:.0f}/{CT_RATED_SECONDARY_A:.0f} A): "
              f"{PICKUP_MIN[k]} - {PICKUP_MAX} kA")
    print(f"Number of Relays     : {NUM_RELAYS}")
    print("Each pair checked at the shared downstream fault current")
    print(f"Coordination Margin  : {COORDINATION_TIME} s")
    print("=======================================\n")