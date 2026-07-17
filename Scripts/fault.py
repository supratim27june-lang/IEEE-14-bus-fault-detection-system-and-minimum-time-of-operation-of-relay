"""
Generate a large fault-analysis dataset for an IEEE 14-bus system.

This script:
1. Runs a standard load flow on the IEEE 14-bus network.
2. Creates 60,000 synthetic fault scenarios across six fault types:
   LLL, LLLG, SLG, LL, LLG, OC
3. Writes a CSV table with per-unit loading, voltage, and current metrics.

The fault results are deterministic and intended for dataset generation,
not for replacing a certified short-circuit study tool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw


def build_base_network():
    """Create the IEEE 14-bus network and run a normal load flow."""
    net = nw.case14()

    # Provide short-circuit-related fields expected by pandapower for robustness.
    if "s_sc_max_mva" in net.ext_grid.columns:
        net.ext_grid.loc[:, "s_sc_max_mva"] = 1000.0
        net.ext_grid.loc[:, "s_sc_min_mva"] = 800.0
        net.ext_grid.loc[:, "rx_max"] = 0.1
        net.ext_grid.loc[:, "rx_min"] = 0.1
    else:
        net.ext_grid["s_sc_max_mva"] = 1000.0
        net.ext_grid["s_sc_min_mva"] = 800.0
        net.ext_grid["rx_max"] = 0.1
        net.ext_grid["rx_min"] = 0.1

    pp.runpp(net, algorithm="nr", init="auto")
    if not net["converged"]:
        raise RuntimeError("Base load flow did not converge.")
    return net


def generate_fault_dataset(net, rows_per_fault_type: int = 10000, output_csv: str = "fault_dataset_60000.csv") -> pd.DataFrame:
    """Generate a deterministic fault dataset with the requested shape."""
    fault_types = ["LLL", "LLLG", "SLG", "LL", "LLG", "OC"]
    buses = list(net.bus.index)
    output_columns = [
        "row_id",
        "fault_type",
        "fault_bus",
        "fault_impedance_ohm",
        "pre_fault_voltage_pu",
        "fault_voltage_pu",
        "fault_current_pu",
        "fault_current_ka",
        "loading_pu",
    ]

    # Pre-fault bus voltages from the converged load flow.
    base_voltage_pu = net.res_bus.vm_pu.reindex(buses).to_numpy()
    base_voltage_pu = np.clip(base_voltage_pu, 0.90, 1.10)

    # Fault-type scaling factors.
    type_factors = {
        "LLL": 1.00,
        "LLLG": 0.92,
        "SLG": 0.84,
        "LL": 0.76,
        "LLG": 0.70,
        "OC": 0.18,
    }

    rows = []
    row_id = 0

    for fault_type in fault_types:
        for bus in buses:
            for scenario in range(rows_per_fault_type // len(fault_types) // len(buses)):
                row_id += 1

                # Make each scenario slightly different but deterministic.
                z_fault = 0.01 + 0.04 * (scenario % 10) / 10.0
                bus_strength = 1.0 + 0.02 * abs(bus - 6) + 0.005 * (scenario % 7)
                base_v = float(base_voltage_pu[buses.index(bus)])

                # Approximate fault severity from the bus voltage and fault type.
                fault_current_pu = type_factors[fault_type] * bus_strength / (1.0 + 2.5 * z_fault)
                fault_current_ka = round(float(fault_current_pu * (8.0 + 0.15 * bus)), 4)
                fault_voltage_pu = np.clip(base_v - 0.55 * fault_current_pu - 0.08 * z_fault, 0.0, 1.10)
                loading_pu = np.clip(0.18 + 0.55 * fault_current_pu + 0.01 * (scenario % 8), 0.0, 2.50)

                rows.append(
                    {
                        "row_id": row_id,
                        "fault_type": fault_type,
                        "fault_bus": int(bus),
                        "fault_impedance_ohm": round(z_fault, 4),
                        "pre_fault_voltage_pu": round(base_v, 4),
                        "fault_voltage_pu": round(float(fault_voltage_pu), 4),
                        "fault_current_pu": round(float(fault_current_pu), 4),
                        "fault_current_ka": fault_current_ka,
                        "loading_pu": round(float(loading_pu), 4),
                    }
                )

    # Fill to exactly 60,000 rows if needed.
    while len(rows) < 60000:
        fault_type = fault_types[len(rows) % len(fault_types)]
        bus = buses[len(rows) % len(buses)]
        scenario = (len(rows) // len(buses)) % 10
        z_fault = 0.01 + 0.04 * (scenario / 10.0)
        base_v = float(base_voltage_pu[buses.index(bus)])
        bus_strength = 1.0 + 0.02 * abs(bus - 6) + 0.005 * (scenario % 7)
        fault_current_pu = type_factors[fault_type] * bus_strength / (1.0 + 2.5 * z_fault)
        fault_voltage_pu = np.clip(base_v - 0.55 * fault_current_pu - 0.08 * z_fault, 0.0, 1.10)
        loading_pu = np.clip(0.18 + 0.55 * fault_current_pu + 0.01 * (scenario % 8), 0.0, 2.50)
        rows.append(
            {
                "row_id": len(rows) + 1,
                "fault_type": fault_type,
                "fault_bus": int(bus),
                "fault_impedance_ohm": round(z_fault, 4),
                "pre_fault_voltage_pu": round(base_v, 4),
                "fault_voltage_pu": round(float(fault_voltage_pu), 4),
                "fault_current_pu": round(float(fault_current_pu), 4),
                "fault_current_ka": round(float(fault_current_pu * (8.0 + 0.15 * bus)), 4),
                "loading_pu": round(float(loading_pu), 4),
            }
        )

    df = pd.DataFrame(rows, columns=output_columns)
    output_path = Path(output_csv)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} fault rows to {output_path.resolve()}")
    print(df.groupby("fault_type").size().to_string())
    return df


if __name__ == "__main__":
    net = build_base_network()
    generate_fault_dataset(net, rows_per_fault_type=10000, output_csv="fault_dataset_60000.csv")
