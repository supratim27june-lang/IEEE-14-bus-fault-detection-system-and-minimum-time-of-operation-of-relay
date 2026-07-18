import streamlit as st
import pandas as pd

from predict import predict_fault
from pso import PSO

# ------------------------------
# Page Config
# ------------------------------

st.set_page_config(
    page_title="Adaptive Relay Optimization",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Machine Learning Assisted Adaptive Relay Coordination")

st.write("---")

# ------------------------------
# Sidebar Inputs
# ------------------------------

st.sidebar.header("Power System Inputs")

fault_bus = st.sidebar.number_input(
    "Fault Bus",
    min_value=0,
    max_value=14,
    value=5
)

fault_current = st.sidebar.number_input(
    "Fault Current (kA)",
    value=5.5
)

fault_impedance = st.sidebar.number_input(
    "Fault Impedance (Ohm)",
    value=0.20
)

loading = st.sidebar.slider(
    "Loading (pu)",
    0.20,
    1.20,
    0.80
)

pre_fault_voltage = st.sidebar.number_input(
    "Pre Fault Voltage (pu)",
    value=1.0
)

fault_voltage = st.sidebar.number_input(
    "Fault Voltage (pu)",
    value=0.65
)

# ------------------------------
# Button
# ------------------------------

if st.button("Predict & Optimize"):

    st.subheader("Machine Learning Prediction")

    fault_type, confidence = predict_fault(
        fault_bus,
        fault_impedance,
        pre_fault_voltage,
        fault_voltage,
        fault_current,
        loading
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Fault",
            fault_type
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

    st.write("---")

    st.subheader("Particle Swarm Optimization")

    optimizer = PSO(
        num_particles=30,
        iterations=10
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

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Optimal TDS",
            round(tds,4)
        )

    with col2:
        st.metric(
            "Pickup Current",
            round(pickup,4)
        )

    with col3:
        st.metric(
            "Operating Time",
            f"{best_time:.4f} sec"
        )

    st.success("Optimization Completed Successfully")
