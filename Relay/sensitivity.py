import pandas as pd
import matplotlib.pyplot as plt

from pso import PSO
from predict import predict_fault

# Current channels that scale with the fault-current magnitude. Voltages,
# impedance, bus and loading stay fixed, so the fault-TYPE signature (the
# relative I2/I0 structure) is preserved while only the magnitude is swept.
CURRENT_COLS = [
    "fault_current_ka", "fault_current_pu",
    "Ia_ka", "Ib_ka", "Ic_ka",
    "I1_ka", "I2_ka", "I0_ka",
    "I1_pu", "I2_pu", "I0_pu",
]

# ---------------------------------------
# Initialize PSO
# ---------------------------------------

optimizer = PSO(
    num_particles=30,
    iterations=40
)

# ---------------------------------------
# Base Operating Condition
# ---------------------------------------
# Take a real measured feature vector from the dataset as the base, then sweep
# its overall current magnitude. This keeps every feature the classifier needs
# physically consistent instead of hand-typing 20+ values.

df = pd.read_csv("fault_dataset_real.csv")
base = df.sample(1, random_state=42).iloc[0].copy()

fault_bus = int(base["fault_bus"])
pre_fault_voltage = float(base["pre_fault_voltage_pu"])
fault_voltage = float(base["fault_voltage_pu"])
loading = float(base["loading_pu"])
fault_impedance = float(base["fault_impedance_ohm"])
base_current = float(base["fault_current_ka"])

# Values to be varied (target fault-current magnitude, kA)
fault_currents = [2,3,4,5,6,7,8,9,10]

results = []

# ---------------------------------------
# Run Sensitivity Analysis
# ---------------------------------------

for fault_current in fault_currents:

    # Scale the whole current vector to the target magnitude.
    scenario = base.copy()
    ratio = fault_current / base_current
    for col in CURRENT_COLS:
        scenario[col] = base[col] * ratio

    fault_type, confidence = predict_fault(scenario)

    best_position, best_time = optimizer.optimize(
        fault_current,
        loading,
        fault_type,
        fault_bus,
        fault_impedance,
        pre_fault_voltage,
        fault_voltage
    )

    tds = best_position[0]
    pickup = best_position[1]

    results.append([
        fault_current,
        fault_type,
        confidence,
        tds,
        pickup,
        best_time
    ])

# ---------------------------------------
# Convert to DataFrame
# ---------------------------------------

columns = [
    "Fault Current (kA)",
    "Predicted Fault",
    "Confidence",
    "Optimal TDS",
    "Pickup Current",
    "Operating Time (s)"
]

df = pd.DataFrame(results, columns=columns)

print(df)

df.to_csv("Sensitivity_Analysis.csv", index=False)

# ---------------------------------------
# Plot 1
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    df["Fault Current (kA)"],
    df["Operating Time (s)"],
    marker='o'
)

plt.grid(True)

plt.xlabel("Fault Current (kA)")
plt.ylabel("Operating Time (s)")
plt.title("Sensitivity Analysis")

plt.show()

# ---------------------------------------
# Plot 2
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    df["Fault Current (kA)"],
    df["Optimal TDS"],
    marker='o'
)

plt.grid(True)

plt.xlabel("Fault Current (kA)")
plt.ylabel("Optimal TDS")
plt.title("TDS Variation")

plt.show()

# ---------------------------------------
# Plot 3
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    df["Fault Current (kA)"],
    df["Pickup Current"],
    marker='o'
)

plt.grid(True)

plt.xlabel("Fault Current (kA)")
plt.ylabel("Pickup Current")
plt.title("Pickup Current Variation")

plt.show()