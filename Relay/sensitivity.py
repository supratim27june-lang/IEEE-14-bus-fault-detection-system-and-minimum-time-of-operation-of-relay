import pandas as pd
import matplotlib.pyplot as plt

from pso import PSO
from predict import predict_fault

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

fault_bus = 5
pre_fault_voltage = 1.0
fault_voltage = 0.65

# Values to be varied
fault_currents = [2,3,4,5,6,7,8,9,10]

loading = 0.80
fault_impedance = 0.20

results = []

# ---------------------------------------
# Run Sensitivity Analysis
# ---------------------------------------

for fault_current in fault_currents:

    fault_type, confidence = predict_fault(
        fault_bus,
        fault_impedance,
        pre_fault_voltage,
        fault_voltage,
        fault_current,
        loading
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