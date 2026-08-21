"""
wilcoxon_significance.py

Pairwise Wilcoxon signed-rank significance testing for the three coordination
methods (Conventional, PSO (no ML), PSO (ML-tuned)) benchmarked in
coordination_benchmark_by_class.py.

Why Wilcoxon signed-rank, not a paired t-test
----------------------------------------------
Operating times across scenarios are not normally distributed (IEC curve
times are strongly right-skewed, and different scenarios span very different
fault-current magnitudes), so the paired t-test's normality assumption is
questionable here. The Wilcoxon signed-rank test only assumes the paired
differences are symmetric about their median, and tests whether one method's
operating time is systematically lower or higher than the other's on the
SAME scenarios -- a much safer assumption for this data.

Pairing basis
--------------
For each sampled fault scenario, all three methods are run and evaluated
exactly as in coordination_benchmark_by_class.py's run_benchmark(): the same
zone-current profile for the pair being compared, one PSO run per method per
scenario. A scenario is kept ONLY if all three methods fully coordinate it
(the "matched subset" already used there for the fair-comparison table) --
this keeps every pairwise test's n identical and avoids biasing the test with
scenarios where one method simply failed to produce a comparable time.

Does NOT modify coordination_benchmark_by_class.py or any other file; reuses
its conventional_settings / pso_optimize / evaluate / classifier plumbing
directly so the settings being compared are identical to the paper's tables.

Usage
------
python wilcoxon_significance.py --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd
from scipy.stats import wilcoxon

# predict_fault (via sklearn's predict_proba) emits a benign UserWarning on
# every call about joblib/delayed config propagation; harmless but floods
# stdout across hundreds of scenarios, so it is silenced here.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from coordination_benchmark_by_class import (
    NO_PICKUP,
    FALLBACK_KA, FALLBACK_MIN_DETECT,
    zone_currents_for_scenario, predict_fault,
    conventional_settings, pso_optimize, evaluate,
)

METHODS = ["Conventional", "PSO (no ML)", "PSO (ML-tuned)"]
ALPHA = 0.05


def collect_matched_times(csv_path, n_scenarios, seed):
    """Run all three methods on the same sampled scenarios and return a dict
    method -> list of operating times, restricted to scenarios where ALL
    THREE methods fully coordinated (matched subset, equal n and order)."""
    df = pd.read_csv(csv_path)
    sample = df.sample(min(n_scenarios, len(df)), random_state=seed)

    matched = {m: [] for m in METHODS}
    n_kept = 0
    for i, (_, sc) in enumerate(sample.iterrows()):
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

        results = {
            "Conventional": (t_conv, c_conv),
            "PSO (no ML)": (t_noml, c_noml),
            "PSO (ML-tuned)": (t_ml, c_ml),
        }

        if all(c and t < NO_PICKUP for t, c in results.values()):
            for m, (t, _c) in results.items():
                matched[m].append(t)
            n_kept += 1

    return matched, n_kept, len(sample)


def run_wilcoxon(matched):
    """Pairwise Wilcoxon signed-rank test on every method pair, on the SAME
    matched-subset scenarios (all three lists share the same length/order,
    so index i in every list is the same underlying scenario)."""
    pairs = [
        ("Conventional", "PSO (no ML)"),
        ("Conventional", "PSO (ML-tuned)"),
        ("PSO (no ML)", "PSO (ML-tuned)"),
    ]
    rows = []
    for a, b in pairs:
        xa, xb = matched[a], matched[b]
        diffs = [x - y for x, y in zip(xa, xb)]
        if not diffs or all(d == 0 for d in diffs):
            rows.append((a, b, len(xa), float("nan"), float("nan"),
                         "identical / no variation -- test not meaningful"))
            continue
        try:
            stat, p = wilcoxon(xa, xb)
        except ValueError as exc:
            rows.append((a, b, len(xa), float("nan"), float("nan"), f"not computable ({exc})"))
            continue
        sorted_diffs = sorted(diffs)
        med_diff = sorted_diffs[len(sorted_diffs) // 2]
        direction = (f"{a} lower" if med_diff < 0 else
                     f"{b} lower" if med_diff > 0 else "no median difference")
        sig = "significant" if p < ALPHA else "not significant"
        rows.append((a, b, len(xa), stat, p,
                     f"{sig} (median diff {med_diff:+.4f}s, {direction})"))
    return rows


def print_report(n_kept, n_sampled, rows):
    print("\n" + "=" * 92)
    print("WILCOXON SIGNED-RANK TEST -- pairwise operating-time comparison (matched scenarios)")
    print("=" * 92)
    print(f"Sampled scenarios           : {n_sampled}")
    print(f"Matched (all 3 coordinated) : {n_kept}  <- n used for every test below")
    print("-" * 92)
    print(f"{'Method A':<18}{'Method B':<18}{'n':>5}{'W stat':>12}{'p-value':>12}   Result")
    print("-" * 92)
    for a, b, n, stat, p, note in rows:
        stat_str = f"{stat:.4f}" if stat == stat else "n/a"
        p_str = f"{p:.4g}" if p == p else "n/a"
        print(f"{a:<18}{b:<18}{n:>5}{stat_str:>12}{p_str:>12}   {note}")
    print("-" * 92)
    print(f"alpha = {ALPHA} (two-sided). H0: paired operating-time differences between the")
    print("two methods are symmetric about zero (no systematic difference).")
    print("=" * 92)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="fault_dataset_hard.csv")
    ap.add_argument("--n", type=int, default=200, help="scenarios to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    matched, n_kept, n_sampled = collect_matched_times(args.csv, args.n, args.seed)
    rows = run_wilcoxon(matched)
    print_report(n_kept, n_sampled, rows)
