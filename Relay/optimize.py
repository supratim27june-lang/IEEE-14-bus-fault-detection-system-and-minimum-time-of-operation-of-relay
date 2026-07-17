import pandas as pd

from pso import PSO
from predict import predict_fault

# -------------------------------------------------
# Pull a REAL fault scenario from the dataset.
#
# The old hand-typed values (fault_impedance_ohm = 0.25, etc.) never occur
# in the simulated data -- real faults cluster near ~0.046 ohm -- so the
# classifier had nothing to match and returned near-random confidence.
# A real row is guaranteed to be in-distribution.
# -------------------------------------------------
df = pd.read_csv("fault_dataset_60000.csv")

# Reproduce a specific case by setting its row_id, or leave None to sample one.
SCENARIO_ROW_ID = None   # e.g. 1234

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
# ML prediction (now on an in-distribution input)
# -------------------------------------------------
fault_type, confidence = predict_fault(
    fault_bus,
    fault_impedance,
    pre_fault_voltage,
    fault_voltage,
    fault_current,
    loading
)

# -------------------------------------------------
# PSO relay-setting optimization
# -------------------------------------------------
optimizer = PSO(
    num_particles=40,
    iterations=60
)

best_position, best_time = optimizer.optimize(
    fault_current,
    loading,
    fault_type,
    fault_bus,
    fault_impedance,
    pre_fault_voltage,
    fault_voltage
)

print()
print(f"Scenario row_id      : {int(scenario['row_id'])}")
print(f"Predicted Fault Type : {fault_type} (confidence: {confidence:.4f})")
print(f"Actual Fault Type    : {true_fault_type}  "
      f"[{'MATCH' if fault_type == true_fault_type else 'MISMATCH'}]")
print()
print("========== OPTIMAL RELAY SETTINGS ==========")
print(f"TDS              : {best_position[0]:.4f}")
print(f"Pickup Current   : {best_position[1]:.4f}")
print(f"Operating Time   : {best_time:.4f} sec")