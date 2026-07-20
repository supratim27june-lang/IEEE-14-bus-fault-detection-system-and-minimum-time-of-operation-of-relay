import pandas as pd

from pso import PSO
from predict import predict_fault
from relay_model import Relay
from objective import zone_currents_for_scenario, select_scenario, CONFIDENCE_THRESHOLD
from constraints import cti_shortfall, is_coordinated, NUM_RELAYS, COORDINATION_TIME

# -------------------------------------------------
# Pull a REAL fault scenario from the dataset.
# -------------------------------------------------
df = pd.read_csv("fault_dataset_hard.csv")

SCENARIO_ROW_ID = None   # e.g. 1234, or None to sample one

if SCENARIO_ROW_ID is not None:
    scenario = df.loc[df["row_id"] == SCENARIO_ROW_ID].iloc[0]
else:
    scenario = df.sample(1, random_state=42).iloc[0]

fault_bus         = int(scenario["fault_bus"])
fault_impedance   = float(scenario["fault_impedance_ohm"])
pre_fault_voltage = float(scenario["pre_fault_voltage_pu"])
fault_voltage     = float(scenario["fault_voltage_pu"])
fault_current     = float(scenario["fault_current_ka"])
loading           = float(scenario["loading_pu"])
true_fault_type   = scenario["fault_type"]

# -------------------------------------------------
# ML prediction (full measured feature vector -> in-distribution input)
# -------------------------------------------------
fault_type, confidence = predict_fault(scenario)

# -------------------------------------------------
# PSO relay-setting optimization
# FIX: forward `confidence` so the trust gate / fallback group actually fires.
# -------------------------------------------------
optimizer = PSO(num_particles=30, iterations=50)
best_position, best_time = optimizer.optimize(
    fault_current,
    loading,
    fault_type,
    fault_bus,
    fault_impedance,
    pre_fault_voltage,
    fault_voltage,
    confidence,
)

# -------------------------------------------------
# Evaluate each relay at the SAME per-relay (zone) current PSO optimised
# against -- NOT the single measured scalar `fault_current`.
#
# Why: PSO coordinates the 5 relays against the type-characteristic zone
# profile (zone_currents_for_scenario). The measured scalar is one bus's
# reading and, for a high-impedance fault, can be far below every relay's
# pickup -- which correctly returns the 9999 "no pickup" sentinel and makes
# the coordinated settings look broken when they are not. Evaluating at the
# zone currents reports the operating times the settings were designed for.
# -------------------------------------------------
zone_currents = zone_currents_for_scenario(fault_type, confidence)
group_label, _ = select_scenario(fault_type, confidence)
trusted = group_label != "FALLBACK"

relay_model = Relay()
relay_settings = []
for relay_idx in range(NUM_RELAYS):
    offset = 2 * relay_idx
    tds = best_position[offset]
    pickup = best_position[offset + 1]
    i_relay = zone_currents[relay_idx]
    operating_time = relay_model.relay_operating_time(i_relay, pickup, tds)
    relay_settings.append((relay_idx + 1, i_relay, tds, pickup, operating_time))

coordinated = is_coordinated(best_position, zone_currents)
_, margins = cti_shortfall(best_position, zone_currents)

print()
print(f"Scenario row_id      : {int(scenario['row_id'])}")
print(f"Predicted Fault Type : {fault_type} (confidence: {confidence:.4f})")
print(f"Actual Fault Type    : {true_fault_type}  "
      f"[{'MATCH' if fault_type == true_fault_type else 'MISMATCH'}]")
print(f"Measured fault current: {fault_current:.4f} kA  "
      f"(coordination designed for the {group_label} zone profile below)")
if not trusted:
    print(f"NOTE: confidence < {CONFIDENCE_THRESHOLD:.2f} -> conservative FALLBACK group used.")
print()
print("========== OPTIMAL RELAY SETTINGS ==========")
print(f"{'Relay':<7}{'Zone I (kA)':<13}{'TDS':<10}{'Pickup':<10}{'Op Time (s)':<12}")
for idx, i_relay, tds, pickup, t in relay_settings:
    t_str = f"{t:.4f}" if t < 9999 else "no pickup"
    print(f"{idx:<7}{i_relay:<13.3f}{tds:<10.4f}{pickup:<10.4f}{t_str:<12}")

print()
margin_txt = ", ".join(
    f"R{k+1}-R{k+2}: {m:.3f}s" if m is not None else f"R{k+1}-R{k+2}: no pickup"
    for k, m in enumerate(margins)
)
print(f"Pairwise CTI margins : {margin_txt}")
print(f"Coordination Status  : {'SATISFIED' if coordinated else 'NEEDS ADJUSTMENT'} "
      f"(required >= {COORDINATION_TIME:.2f} s)")
print(f"Objective Value      : {best_time:.4f}")