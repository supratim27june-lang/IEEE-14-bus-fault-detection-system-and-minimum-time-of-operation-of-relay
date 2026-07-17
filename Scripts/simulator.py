"""
IEEE 14-BUS SYSTEM — LOAD FLOW ANALYSIS + VISUALIZATION (pandapower)
=====================================================================
Run:
    python simulator.py

Saves 2 PNG charts to the current folder AND auto-opens them with your
default image viewer (Windows). No GUI backend required.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")            # always works, no display needed to SAVE
import matplotlib.pyplot as plt

import pandapower as pp
import pandapower.networks as nw
import pandas as pd

pd.set_option("display.float_format", lambda x: f"{x:9.4f}")
pd.set_option("display.width", 120)


def run_loadflow():
    net = nw.case14()

    print("=" * 70)
    print("IEEE 14-BUS SYSTEM — NETWORK SUMMARY")
    print("=" * 70)
    print(net)

    pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=True)
    if not net["converged"]:
        raise RuntimeError("Power flow did NOT converge.")
    print("\nPower flow converged successfully (Newton-Raphson).\n")

    bus_res = net.res_bus.copy()
    bus_res.index.name = "bus"
    bus_res["vn_kv"] = net.bus["vn_kv"]
    bus_res = bus_res[["vn_kv", "vm_pu", "va_degree", "p_mw", "q_mvar"]]
    print("=" * 70)
    print("BUS RESULTS")
    print("=" * 70)
    print(bus_res.to_string())

    line_res = net.res_line.copy()
    line_res.index.name = "line"
    line_res["from_bus"] = net.line["from_bus"]
    line_res["to_bus"] = net.line["to_bus"]
    line_res = line_res[["from_bus", "to_bus", "p_from_mw", "q_from_mvar",
                          "pl_mw", "ql_mvar", "loading_percent"]]
    print("\n" + "=" * 70)
    print("LINE RESULTS")
    print("=" * 70)
    print(line_res.to_string())

    print("\n" + "=" * 70)
    print("GENERATION")
    print("=" * 70)
    print("\nExternal grid (slack bus):")
    print(net.res_ext_grid.to_string())
    if len(net.gen):
        gen_res = net.res_gen.copy()
        gen_res["bus"] = net.gen["bus"].values
        print("\nGenerators (PV buses):")
        print(gen_res[["bus", "p_mw", "q_mvar", "vm_pu", "va_degree"]].to_string())

    total_gen_p = net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum()
    total_load_p = net.res_load.p_mw.sum()
    loss_p = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()

    print("\n" + "=" * 70)
    print("SYSTEM SUMMARY")
    print("=" * 70)
    print(f"  Total generation : {total_gen_p:8.3f} MW")
    print(f"  Total load       : {total_load_p:8.3f} MW")
    print(f"  Total losses     : {loss_p:8.3f} MW")
    print(f"  Max line loading : {net.res_line.loading_percent.max():6.2f} %")
    print(f"  Min bus voltage  : {net.res_bus.vm_pu.min():6.4f} p.u.")
    print(f"  Max bus voltage  : {net.res_bus.vm_pu.max():6.4f} p.u.")
    print("=" * 70)

    try:
        with pd.ExcelWriter("ieee14_results.xlsx") as writer:
            bus_res.to_excel(writer, sheet_name="bus")
            line_res.to_excel(writer, sheet_name="line")
        print("\nNumerical results saved to 'ieee14_results.xlsx'")
    except Exception as e:
        print(f"\n(Excel export skipped: {e})")

    return net


def open_file(path):
    """Open a file with the OS default viewer. Works on Windows/Mac/Linux."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except Exception as e:
        print(f"Could not auto-open {path}: {e}")


def make_status_plot(net):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f7fff7")
    ax.set_facecolor("#f7fff7")
    ax.axis("off")

    # Large checkmark / status indicator
    ax.text(
        0.5, 0.72, "✓",
        ha="center", va="center",
        fontsize=64, color="#2e7d32", fontweight="bold"
    )
    ax.text(
        0.5, 0.50, "SIMULATION OK",
        ha="center", va="center",
        fontsize=24, color="#1b5e20", fontweight="bold"
    )

    metrics = [
        ("Converged", "Yes", "#2e7d32"),
        ("Max loading", f"{net.res_line.loading_percent.max():.1f}%", "#1565c0"),
        ("Min voltage", f"{net.res_bus.vm_pu.min():.4f} p.u.", "#1565c0"),
        ("Max voltage", f"{net.res_bus.vm_pu.max():.4f} p.u.", "#1565c0"),
    ]

    for idx, (label, value, color) in enumerate(metrics):
        y_pos = 0.30 - idx * 0.07
        ax.text(0.28, y_pos, label, ha="left", va="center", fontsize=12, color="#263238")
        ax.text(0.62, y_pos, value, ha="left", va="center", fontsize=12, color=color, fontweight="bold")

    ax.text(
        0.5, 0.08,
        "Newton-Raphson load flow completed successfully",
        ha="center", va="center",
        fontsize=10, color="#2e7d32"
    )

    plt.tight_layout()
    path = os.path.abspath("ieee14_simulation_status.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"\nSaved status image: {path}")
    open_file(path)
    return path


def make_plots(net):
    saved_files = []

    # ---------- PLOT 1: voltage profile ----------
    vm = net.res_bus.vm_pu
    lo, hi = 0.94, 1.06
    colors = ["#2e7d32" if lo <= v <= hi else "#c62828" for v in vm]
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar([str(i) for i in net.bus.index], vm, color=colors, edgecolor="black")
    ax1.axhline(hi, color="red", ls="--", lw=1, label=f"Upper {hi}")
    ax1.axhline(lo, color="red", ls="--", lw=1, label=f"Lower {lo}")
    ax1.axhline(1.0, color="grey", ls=":", lw=1)
    ax1.set_ylim(0.90, 1.12)
    ax1.set_xlabel("Bus")
    ax1.set_ylabel("Voltage (p.u.)")
    ax1.set_title("IEEE 14-Bus — Voltage Profile")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path1 = os.path.abspath("ieee14_voltage_profile.png")
    fig1.savefig(path1, dpi=140)
    plt.close(fig1)
    saved_files.append(path1)

    # ---------- PLOT 2: line loading ----------
    load = net.res_line.loading_percent
    labels = [f"{net.line.from_bus[i]}-{net.line.to_bus[i]}" for i in net.line.index]
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.bar(labels, load, color="#1565c0", edgecolor="black")
    ax2.axhline(100, color="red", ls="--", lw=1, label="100% thermal limit")
    ax2.set_xlabel("Line (from-to bus)")
    ax2.set_ylabel("Loading (%)")
    ax2.set_title("IEEE 14-Bus — Line Loading")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path2 = os.path.abspath("ieee14_line_loading.png")
    fig2.savefig(path2, dpi=140)
    plt.close(fig2)
    saved_files.append(path2)

    # ---------- PLOT 3: simple network topology (built-in, safe) ----------
    try:
        import pandapower.plotting as pplot
        fig3 = plt.figure(figsize=(9, 7))
        pplot.simple_plot(net, respect_switches=True, show_plot=False)
        path3 = os.path.abspath("ieee14_network.png")
        plt.savefig(path3, dpi=140)
        plt.close()
        saved_files.append(path3)
    except Exception as e:
        print(f"(Network diagram skipped: {e})")

    print("\nSaved plot files:")
    for f in saved_files:
        print("  -", f)

    print("\nOpening images now...")
    for f in saved_files:
        open_file(f)


if __name__ == "__main__":
    net = run_loadflow()
    make_status_plot(net)
    make_plots(net)