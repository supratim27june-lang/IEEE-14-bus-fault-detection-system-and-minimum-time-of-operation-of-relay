"""
coordination_benchmark_by_class.py

Independent, standalone script that fixes Major Concern 4 (coordination rate
must be broken down by TRUE fault class, not reported only in aggregate).

Does NOT modify generalized_relay_pso.py or any other existing file.

Why this is a NEW script rather than a patch to generalized_relay_pso.py
-------------------------------------------------------------------------
generalized_relay_pso.py implements a single-relay, 2-variable optimizer
(pickup, TDS only). It has no 5-relay decision vector, no CTI constraint, no
classifier/confidence gate, and no conventional baseline -- it structurally
cannot produce the Conventional / PSO-no-ML / PSO-ML-tuned comparison in
Tables 6-10 of the manuscript. There is therefore nothing in it to preserve;
this script reimplements the actual 5-relay benchmark from the project's
existing coordination modules (objective.py, pso.py, constraints.py,
relay_model.py) and adds the missing per-class breakdown on top.

What this fixes
----------------
Section 4.6 reports ONE aggregate coordination rate per method (e.g. 29.5%,
40.0%, 45.5%). Section 4.8 separately shows WHY it's that low for OC (real
per-relay OC currents fall below the pickup floor at R3-R5). Nothing in the
paper currently connects the two. This script computes coordination rate and
operating time SEPARATELY for each true fault class, so the reader can see
directly whether the low aggregate is driven by OC (as hypothesized) or is a
general problem across all classes.

Reference currents
-------------------
Uses the manuscript's own published Table 4 values (min-generation case,
already voltage-referred and ordering-corrected) -- NOT the older placeholder
table from earlier development. If Table 4 has since been revised, update
TABLE4_MIN_GEN_KA below to match.
"""

from __future__ import annotations

import random
import statistics as stats
from collections import defaultdict

import pandas as pd
import joblib

from relay_model import Relay
from constraints import (
    NUM_RELAYS, COORDINATION_TIME, cti_shortfall, is_coordinated,
    TDS_MIN, TDS_MAX, PICKUP_MIN, PICKUP_MAX,
)
from features import FEATURES

# -----------------------------------------------------------------------
# Relay parameter bounds are now imported directly from constraints.py.
# constraints.py is the single source of truth: PICKUP_MIN is a per-relay
# list derived from each relay's own CT (current transformer) secondary
# rating (1.2-1.5x the CT's rated secondary current, per IEC/ANSI practice),
# not one global floor -- see constraints.py for the per-relay CT ratios and
# why they were chosen. (Earlier this script hardcoded its own local scalar
# because constraints.py's PICKUP_MIN was stale at the time; that is no
# longer true, so the local override has been removed to keep both scripts
# consistent.)
# -----------------------------------------------------------------------

relay = Relay()
NO_PICKUP = 9999.0
CONFIDENCE_THRESHOLD = 0.70  # replace with your swept/justified value if different

# ---------------------------------------------------------------------------
# Manuscript Table 4 (min-generation, voltage-referred, ordering-corrected),
# per-relay reference currents in kA. R1..R5 order matches the paper.
# ---------------------------------------------------------------------------
TABLE4_MIN_GEN_KA = {
    "LLL": [2.808, 1.482, 4.444, 0.410, 0.512],
    "LL":  [2.431, 1.178, 2.571, 0.355, 0.444],
    "SLG": [2.034, 1.020, 2.227, 0.231, 0.359],
    "OC":  [1.133, 0.408, 0.120, 0.073, 0.043],
}
FALLBACK_KA = [min(TABLE4_MIN_GEN_KA[t][k] for t in TABLE4_MIN_GEN_KA)
               for k in range(NUM_RELAYS)]
TYPE_MIN_DETECT = {t: min(v) for t, v in TABLE4_MIN_GEN_KA.items()}
FALLBACK_MIN_DETECT = min(TYPE_MIN_DETECT.values())


def zone_currents_for_scenario(fault_type, confidence):
    if confidence >= CONFIDENCE_THRESHOLD and fault_type in TABLE4_MIN_GEN_KA:
        return TABLE4_MIN_GEN_KA[fault_type], TYPE_MIN_DETECT[fault_type]
    return FALLBACK_KA, FALLBACK_MIN_DETECT


# ---------------------------------------------------------------------------
# Classifier (loads the project's existing trained model; does not retrain)
# ---------------------------------------------------------------------------
_model = joblib.load("rf_fault_classifier.pkl")


def predict_fault(row):
    X = pd.DataFrame([row])[FEATURES]
    pred = _model.predict(X)[0]
    conf = float(_model.predict_proba(X)[0].max())
    return pred, conf


# ---------------------------------------------------------------------------
# Shared evaluator -- identical for all three methods, so no method gets a
# measurement advantage.
# ---------------------------------------------------------------------------
def evaluate(position, zone_currents):
    total = sum(
        relay.relay_operating_time(zone_currents[k], position[2 * k + 1], position[2 * k])
        for k in range(NUM_RELAYS)
    )
    return total, is_coordinated(position, zone_currents)


# ---------------------------------------------------------------------------
# METHOD 1: Conventional (analytic bottom-up CTI grading, no optimization)
#
# Pickup is set from the scenario's LOAD current plus a standard margin --
# NOT as a fixed fraction of the relay's own zone fault current. The
# manuscript (Section 3.8) explicitly reports that a fixed-fraction-of-fault-
# current construction was found to make the baseline a scenario-independent
# constant, and was replaced with load-based pickup for exactly this reason.
# It also matters physically here: a backup relay's pickup must stay low
# enough to still detect a much SMALLER downstream zone current (e.g. R3's
# own zone current is a local maximum at the step-down transformer, R4's is
# far smaller) -- sizing pickup from the relay's own zone current alone
# breaks that requirement.
# ---------------------------------------------------------------------------
def conventional_settings(zone_currents, loading):
    position = [0.0] * (2 * NUM_RELAYS)
    pickups = []
    for k in range(NUM_RELAYS):
        load_based = 1.25 * loading * (0.85 ** k)
        pu = min(max(load_based, PICKUP_MIN[k]), PICKUP_MAX)
        pickups.append(pu)

    idx = NUM_RELAYS - 1
    position[2 * idx], position[2 * idx + 1] = TDS_MIN, pickups[idx]

    for k in range(NUM_RELAYS - 2, -1, -1):
        i_shared = zone_currents[k + 1]
        t_down = relay.relay_operating_time(i_shared, pickups[k + 1], position[2 * (k + 1)])
        if t_down >= NO_PICKUP:
            position[2 * k] = TDS_MIN
        else:
            ratio = i_shared / pickups[k]
            tds = (t_down + COORDINATION_TIME) * (ratio ** 0.02 - 1.0) / 0.14 if ratio > 1 else TDS_MAX
            position[2 * k] = min(max(tds, TDS_MIN), TDS_MAX)
        position[2 * k + 1] = pickups[k]
    return position


# ---------------------------------------------------------------------------
# METHOD 2 & 3: PSO (lightweight, self-contained -- no dependency on pso.py's
# particle-class internals, so this script has zero coupling to files that
# might change independently)
# ---------------------------------------------------------------------------
def _constructive_seed(zone_currents, loading):
    """Feasible-leaning starting vector, mirroring the project's validated
    pso.py warm-start: load-based pickups, TDS solved so operating time
    increases by roughly one CTI per relay walking upstream. Without this,
    random initialization alone was found (empirically, below) to fail to
    reach the tightly-constrained feasible region within a practical
    iteration budget."""
    pos = []
    for k in range(NUM_RELAYS):
        pickup = max(PICKUP_MIN[k] + 0.1 * PICKUP_MIN[k], 1.25 * loading * (0.85 ** k))
        target = 0.25 + (COORDINATION_TIME + 0.1) * (NUM_RELAYS - 1 - k)
        ratio = zone_currents[k] / pickup
        tds = target * (ratio ** 0.02 - 1) / 0.14 if ratio > 1 else 0.5
        tds = min(max(tds, TDS_MIN), TDS_MAX)
        pos += [tds, pickup]
    return pos


def pso_optimize(zone_currents, min_detect, loading, n=40, iters=150, seed=None):
    if seed is not None:
        random.seed(seed)
    dim = 2 * NUM_RELAYS
    lo, hi = [], []
    for k in range(NUM_RELAYS):
        lo += [TDS_MIN, PICKUP_MIN[k]]
        hi += [TDS_MAX, PICKUP_MAX]

    def fitness(pos):
        cti, _ = cti_shortfall(pos, zone_currents)
        bnd = sum(max(0.0, TDS_MIN - pos[2*k]) + max(0.0, pos[2*k] - TDS_MAX) +
                  max(0.0, PICKUP_MIN[k] - pos[2*k+1]) + max(0.0, pos[2*k+1] - PICKUP_MAX)
                  for k in range(NUM_RELAYS))
        ceiling = (2.0 / 3.0) * min_detect
        pick = 0.0
        for k in range(NUM_RELAYS):
            pu = pos[2 * k + 1]
            if pu > ceiling:
                pick += (pu - ceiling)
            floor = 1.25 * loading * (0.85 ** k)
            if pu < floor:
                pick += (floor - pu)
        tot = evaluate(pos, zone_currents)[0]
        return 1e4 * cti + 1e3 * bnd + 1e2 * pick + 20.0 * tot  # time weight raised for fair measurement

    X = [[random.uniform(lo[d], hi[d]) for d in range(dim)] for _ in range(n)]
    X[0] = _constructive_seed(zone_currents, loading)  # warm-start (critical -- see docstring above)
    V = [[0.0] * dim for _ in range(n)]
    pb, pbv = [p[:] for p in X], [fitness(p) for p in X]
    g = min(range(n), key=lambda i: pbv[i])
    gb, gbv = pb[g][:], pbv[g]
    for it in range(iters):
        w = 0.9 - 0.5 * it / iters
        for i in range(n):
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                V[i][d] = w * V[i][d] + 1.5 * r1 * (pb[i][d] - X[i][d]) + 1.5 * r2 * (gb[d] - X[i][d])
                vmax = 0.2 * (hi[d] - lo[d])
                V[i][d] = max(-vmax, min(V[i][d], vmax))
                X[i][d] = min(max(X[i][d] + V[i][d], lo[d]), hi[d])
            v = fitness(X[i])
            if v < pbv[i]:
                pb[i], pbv[i] = X[i][:], v
                if v < gbv:
                    gb, gbv = X[i][:], v
    return gb


# ---------------------------------------------------------------------------
# Benchmark driver with PER-CLASS breakdown (the Major Concern 4 fix)
# ---------------------------------------------------------------------------
def run_benchmark(csv_path, n_scenarios, seed=42):
    df = pd.read_csv(csv_path)
    sample = df.sample(min(n_scenarios, len(df)), random_state=seed)

    methods = ["Conventional", "PSO (no ML)", "PSO (ML-tuned)"]
    # by_class[method][fault_type] = [n_coordinated, n_total, [times...]]
    by_class = {m: defaultdict(lambda: [0, 0, []]) for m in methods}

    for i, (_, sc) in enumerate(sample.iterrows()):
        true_type = sc["fault_type"]           # KEY by TRUE type, not predicted
        loading = float(sc["loading_pu"])

        pred_type, confidence = predict_fault(sc)
        zc_ml, min_ml = zone_currents_for_scenario(pred_type, confidence)
        zc_fb, min_fb = FALLBACK_KA, FALLBACK_MIN_DETECT

        pos_conv = conventional_settings(zc_ml, loading)
        t_conv, c_conv = evaluate(pos_conv, zc_ml)

        pos_noml = pso_optimize(zc_fb, min_fb, loading, seed=seed + i)
        t_noml, c_noml = evaluate(pos_noml, zc_fb)

        pos_ml = pso_optimize(zc_ml, min_ml, loading, seed=seed + i + 100000)
        t_ml, c_ml = evaluate(pos_ml, zc_ml)

        for m, (t, c) in zip(methods, [(t_conv, c_conv), (t_noml, c_noml), (t_ml, c_ml)]):
            entry = by_class[m][true_type]
            entry[1] += 1
            if c:
                entry[0] += 1
                if t < NO_PICKUP:
                    entry[2].append(t)

    return by_class, methods, len(sample)


def print_table(by_class, methods, n):
    all_types = sorted({t for m in methods for t in by_class[m]})

    print("\n" + "=" * 90)
    print(f"COORDINATION BENCHMARK BY FAULT CLASS (n={n} scenarios, min-generation)")
    print("=" * 90)
    header = f"{'Fault type':<12}{'n':>6}"
    for m in methods:
        header += f"{m + ' coord%':>20}"
    print(header)
    print("-" * 90)

    agg_coord = {m: [0, 0] for m in methods}
    agg_times = {m: [] for m in methods}

    for ft in all_types:
        row = f"{ft:<12}"
        n_this = by_class[methods[0]][ft][1]
        row += f"{n_this:>6}"
        for m in methods:
            c, tot, _ = by_class[m][ft]
            pct = c / tot * 100 if tot else 0.0
            row += f"{pct:>19.1f}%"
            agg_coord[m][0] += c
            agg_coord[m][1] += tot
            agg_times[m].extend(by_class[m][ft][2])
        print(row)

    print("-" * 90)
    row = f"{'Overall':<12}{n:>6}"
    for m in methods:
        c, tot = agg_coord[m]
        pct = c / tot * 100 if tot else 0.0
        row += f"{pct:>19.1f}%"
    print(row)
    print("=" * 90)

    print("\nMean operating time (s) by fault class, among scenarios that fully coordinated:")
    header = f"{'Fault type':<12}"
    for m in methods:
        header += f"{m:>20}"
    print(header)
    for ft in all_types:
        row = f"{ft:<12}"
        for m in methods:
            times = by_class[m][ft][2]
            row += f"{(stats.mean(times) if times else float('nan')):>20.3f}"
        print(row)

    # Explicit check for the OC hypothesis (Section 4.8 link)
    if "OC" in all_types:
        non_oc = [t for t in all_types if t != "OC"]
        print("\n--- OC vs. short-circuit-class coordination rate (Concern 4 check) ---")
        for m in methods:
            oc_c, oc_n, _ = by_class[m]["OC"]
            sc_c = sum(by_class[m][t][0] for t in non_oc)
            sc_n = sum(by_class[m][t][1] for t in non_oc)
            oc_pct = oc_c / oc_n * 100 if oc_n else float("nan")
            sc_pct = sc_c / sc_n * 100 if sc_n else float("nan")
            print(f"  {m:<18}: OC={oc_pct:.1f}%  (LLL+LL+SLG)={sc_pct:.1f}%  "
                  f"gap={sc_pct - oc_pct:+.1f} pp")


if __name__ == "__main__":
    by_class, methods, n = run_benchmark("fault_dataset_hard.csv", n_scenarios=200, seed=42)
    print_table(by_class, methods, n)