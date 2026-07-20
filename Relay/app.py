import numpy as np
import pandas as pd
import streamlit as st

from predict import predict_fault
from pso import PSO
from relay_model import Relay
from constraints import COORDINATION_TIME, NUM_RELAYS, cti_shortfall, is_coordinated
from objective import CONFIDENCE_THRESHOLD, select_scenario, zone_currents_for_scenario

relay = Relay()

st.set_page_config(page_title="Adaptive Relay Optimization", page_icon="⚡", layout="wide")
st.title("⚡ Machine Learning Assisted Adaptive 5-Relay Coordination")
st.write("---")

# The classifier is trained on the full physical feature vector (per-phase and
# sequence currents + per-phase voltages + operating conditions). We only ask
# the user for the 5 headline quantities a relay/DFR readout would show; the
# remaining measurements (Ia/Ib/Ic, I1/I2/I0, Va/Vb/Vc, loading) are derived
# from the closest matching event in the training distribution, rescaled to
# match the entered current/voltage magnitudes exactly.
DATA_CSV = "fault_dataset_hard.csv"
NEIGHBOR_COLS = ["fault_impedance_ohm", "fault_current_ka", "fault_voltage_pu", "pre_fault_voltage_pu"]
DERIVED_COLS = ["Ia_ka", "Ib_ka", "Ic_ka", "I1_ka", "I2_ka", "I0_ka", "Va_pu", "Vb_pu", "Vc_pu"]


@st.cache_data
def load_reference():
    return pd.read_csv(DATA_CSV)


def nearest_reference_row(df, fault_bus, fault_impedance, fault_current, fault_voltage, pre_fault_voltage):
    """Closest training-distribution event to the typed inputs.

    Prefers events at the same bus (same network location -> same feeder
    impedance/attenuation pattern); falls back to the whole dataset if the
    bus has no rows for some reason.
    """
    candidates = df[df["fault_bus"] == fault_bus]
    if candidates.empty:
        candidates = df

    query = np.array([fault_impedance, fault_current, fault_voltage, pre_fault_voltage])
    mu = df[NEIGHBOR_COLS].mean().to_numpy()
    sd = df[NEIGHBOR_COLS].std().replace(0, 1).to_numpy()
    z_candidates = (candidates[NEIGHBOR_COLS].to_numpy() - mu) / sd
    z_query = (query - mu) / sd
    dist = np.sqrt(((z_candidates - z_query) ** 2).sum(axis=1))
    return candidates.iloc[dist.argmin()]


def build_scenario_row(df, fault_bus, fault_impedance, fault_current, fault_voltage, pre_fault_voltage):
    """Full 16-feature row for the classifier, sourced from the typed
    headline values plus a physically-consistent derivation of the rest."""
    nearest = nearest_reference_row(
        df, fault_bus, fault_impedance, fault_current, fault_voltage, pre_fault_voltage
    )

    current_ratio = fault_current / max(nearest["fault_current_ka"], 1e-6)
    voltage_ratio = fault_voltage / max(nearest["fault_voltage_pu"], 1e-6)

    row = nearest.to_dict()
    row["fault_bus"] = fault_bus
    row["fault_impedance_ohm"] = fault_impedance
    row["pre_fault_voltage_pu"] = pre_fault_voltage
    row["fault_voltage_pu"] = fault_voltage
    row["fault_current_ka"] = fault_current
    row["fault_current_pu"] = nearest["fault_current_pu"] * current_ratio
    for col in ["Ia_ka", "Ib_ka", "Ic_ka", "I1_ka", "I2_ka", "I0_ka"]:
        row[col] = max(0.0, nearest[col] * current_ratio)
    for col in ["Va_pu", "Vb_pu", "Vc_pu"]:
        row[col] = float(np.clip(nearest[col] * voltage_ratio, 0.0, 1.2))

    return pd.Series(row), nearest


df = load_reference()

st.sidebar.header("Fault Scenario")
st.sidebar.caption("Enter the readings a relay/DFR captured for this event.")

fault_bus = st.sidebar.selectbox(
    "Fault bus", options=sorted(df["fault_bus"].unique().tolist()), index=0
)
fault_impedance = st.sidebar.number_input(
    "Fault impedance (Ω)", min_value=0.0, value=0.5, step=0.1, format="%.3f"
)
fault_current = st.sidebar.number_input(
    "Fault current (kA)", min_value=0.01, value=5.0, step=0.1, format="%.3f"
)
fault_voltage = st.sidebar.number_input(
    "Fault (retained) voltage (pu)", min_value=0.0, max_value=1.2, value=0.5, step=0.01, format="%.3f"
)
pre_fault_voltage = st.sidebar.number_input(
    "Pre-fault voltage (pu)", min_value=0.5, max_value=1.2, value=1.0, step=0.01, format="%.3f"
)

scenario, nearest = build_scenario_row(
    df, fault_bus, fault_impedance, fault_current, fault_voltage, pre_fault_voltage
)
loading = float(scenario["loading_pu"])

with st.sidebar.expander("Derived measurements (from closest reference event)"):
    st.write(
        f"I1/I2/I0: {scenario['I1_ka']:.2f} / {scenario['I2_ka']:.2f} / "
        f"{scenario['I0_ka']:.2f} kA"
    )
    st.write(
        f"Ia/Ib/Ic: {scenario['Ia_ka']:.2f} / {scenario['Ib_ka']:.2f} / "
        f"{scenario['Ic_ka']:.2f} kA"
    )
    st.write(
        f"Va/Vb/Vc: {scenario['Va_pu']:.2f} / {scenario['Vb_pu']:.2f} / "
        f"{scenario['Vc_pu']:.2f} pu"
    )
    st.write(f"Loading: {loading:.2f} pu (from nearest reference event, bus {int(nearest['fault_bus'])})")

if st.button("Predict & Optimize"):

    st.subheader("Machine Learning Prediction")
    fault_type, confidence = predict_fault(scenario)
    c1, c2 = st.columns(2)
    c1.metric("Predicted Fault", fault_type)
    c2.metric("Confidence", f"{confidence * 100:.2f}%")

    group_label, _min_detect = select_scenario(fault_type, confidence)
    if group_label != "FALLBACK":
        st.info(f"Confidence ≥ {CONFIDENCE_THRESHOLD*100:.0f}% → using the **{group_label}** setting group.")
    else:
        st.warning(f"Confidence < {CONFIDENCE_THRESHOLD*100:.0f}% → using the **conservative fallback** group.")

    st.write("---")
    st.subheader("5-Relay Coordination Optimization")

    optimizer = PSO(num_particles=40, iterations=200)
    best_position, best_value = optimizer.optimize(
        fault_current, loading, fault_type, fault_bus,
        fault_impedance, pre_fault_voltage, fault_voltage, confidence,
    )

    zone_currents = zone_currents_for_scenario(fault_type, confidence)
    _, margins = cti_shortfall(best_position, zone_currents)  # per-pair margins, correct definition

    rows = []
    for k in range(NUM_RELAYS):
        tds, pickup = best_position[2 * k], best_position[2 * k + 1]
        i_zone = zone_currents[k]
        t = relay.relay_operating_time(i_zone, pickup, tds) if pickup < i_zone else float("inf")
        rows.append({
            "Relay": k + 1,
            "Zone Fault I (kA)": round(i_zone, 3),
            "TDS": round(tds, 4),
            "Pickup Current": round(pickup, 4),
            "Operating Time (s)": round(t, 4) if t != float("inf") else "no pickup",
        })

    st.metric("Relay 1 (primary) operating time", rows[0]["Operating Time (s)"] if isinstance(rows[0]["Operating Time (s)"], (int, float)) else rows[0]["Operating Time (s)"])

    coordinated = is_coordinated(best_position, zone_currents)
    c1, c2, c3 = st.columns(3)
    c1.metric("Objective Value", f"{best_value:.4f}")
    c2.metric("Coordination Margin", f"≥ {COORDINATION_TIME:.2f} s")
    c3.metric("Coordination Status", "Satisfied" if coordinated else "Needs adjustment")

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # show the actual per-pair margins so the status is auditable
    margin_txt = ", ".join(
        f"R{k+1}-R{k+2}: {m:.3f}s" if m is not None else f"R{k+1}-R{k+2}: no pickup"
        for k, m in enumerate(margins)
    )
    st.caption(f"Pairwise backup–primary margins (at shared downstream current): {margin_txt}")

    if coordinated:
        st.success("Coordination satisfied — every primary/backup pair meets the CTI margin.")
    else:
        st.error("Coordination not satisfied — a primary/backup pair is below the CTI margin.")
