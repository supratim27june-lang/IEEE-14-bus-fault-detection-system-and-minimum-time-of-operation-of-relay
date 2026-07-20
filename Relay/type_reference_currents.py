"""
type_reference_currents.py

Replace the hand-picked TYPE_REF_CURRENT / TYPE_MIN_CURRENT placeholder table in
objective.py with REAL values from pandapower.shortcircuit.calc_sc (IEC 60909).

Scope of this script (deliberately limited, per your plan)
------------------------------------------------------------
This computes NETWORK-WIDE characteristic currents per fault type -- it does
NOT yet place 5 relays on specific lines. That comes next, once you pick the
relay path; this script is the "get the currents right" step before it.

Nothing here gets thrown away once you do placement: today it reads bus-level
fault currents (calc_sc(..., bus=<every bus>)); after placement, the same
calc_sc call just targets each relay's specific downstream bus and reads the
current through that relay's line (net.res_line_sc) instead of the network-wide
bus value. The fault-type/case loop structure is identical either way.

Why BOLTED duty (Zf = 0), not the noisy classifier dataset
------------------------------------------------------------
fault_dataset_hard.csv deliberately has fault-impedance variation, CT
saturation, and measurement noise -- that's what makes classification
realistic. Coordination reference currents are a different physical quantity:
they are what protection studies actually use to set relay TDS/pickup, and the
standard practice (IEC 60909 style) is to coordinate against the BOLTED
short-circuit duty at maximum and minimum generation -- the two extremes that
bound how much current a relay could see. Using the noisy sensor-side dataset
for this would inject measurement noise into a design calculation where it
doesn't belong, and we already found it destroys the physical fault-type
ordering (LLL stopped being the strongest fault). Bolted max/min duty is both
more standard AND avoids that problem.

Output
------
Prints TYPE_REF_CURRENT and TYPE_MIN_CURRENT as ready-to-paste Python dicts,
for both case="max" and case="min", plus the sanity checks that catch the two
failure modes we've hit before (wrong ordering, unphysical magnitude).

REQUIRES pandapower. Untested against your live network -- run the printed
sanity checks before pasting anything into objective.py.
"""

from __future__ import annotations

import math
import numpy as np
import pandapower as pp
import pandapower.networks as nw
import pandapower.shortcircuit as sc


# ---------------------------------------------------------------------------
# Self-contained: build_base_network() and SC_FAULTS are copied here rather
# than imported from fault_hard.py, because that file lives in a different
# folder (Scripts/ vs Relay/ in your project) and a cross-folder import is
# fragile -- it breaks the moment either file moves. This copy is IDENTICAL
# to the one in fault_hard.py so the coordination currents come from the
# exact same network model as your classifier dataset. If you ever change
# build_base_network in fault_hard.py (e.g. different zero-sequence data),
# mirror the change here too, or this script silently drifts out of sync.
# ---------------------------------------------------------------------------
BASE_MVA = 100.0
SC_FAULTS = {"LLL": "3ph", "LL": "2ph", "SLG": "1ph"}


def build_base_network():
    net = nw.case14()

    net.ext_grid["s_sc_max_mva"] = 1000.0
    net.ext_grid["s_sc_min_mva"] = 800.0
    net.ext_grid["rx_max"] = 0.1
    net.ext_grid["rx_min"] = 0.1
    net.ext_grid["x0x_max"] = 1.0
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_min"] = 1.0
    net.ext_grid["r0x0_min"] = 0.1

    if not net.gen.empty:
        cos_phi = 0.85
        net.gen["vn_kv"] = net.bus.loc[net.gen.bus.values, "vn_kv"].values
        net.gen["sn_mva"] = (net.gen["max_p_mw"].fillna(100.0) / cos_phi).round(1)
        net.gen["cos_phi"] = cos_phi
        net.gen["xdss_pu"] = 0.20
        net.gen["rdss_ohm"] = (0.02 * net.gen["vn_kv"] ** 2 / net.gen["sn_mva"]).round(6)
        net.gen["pg_percent"] = 0.0
        net.gen["power_station_trafo"] = np.nan

    if not net.trafo.empty:
        net.trafo["vector_group"] = "Dyn"
        net.trafo["vk0_percent"] = net.trafo["vk_percent"]
        net.trafo["vkr0_percent"] = net.trafo["vkr_percent"]
        net.trafo["mag0_percent"] = 100.0
        net.trafo["mag0_rx"] = 0.0
        net.trafo["si0_hv_partial"] = 0.9

    if "r0_ohm_per_km" not in net.line.columns or net.line["r0_ohm_per_km"].isna().any():
        net.line["r0_ohm_per_km"] = net.line["r_ohm_per_km"] * 3.0
        net.line["x0_ohm_per_km"] = net.line["x_ohm_per_km"] * 3.0
        net.line["c0_nf_per_km"] = net.line["c_nf_per_km"] * 0.6
        net.line["endtemp_degree"] = 80.0

    pp.runpp(net, algorithm="nr", init="auto")
    if not net["converged"]:
        raise RuntimeError("Base load flow did not converge.")
    return net


C_FACTOR = {"max": 1.10, "min": 1.00}     # IEC 60909 voltage factor per case
MIN_CURRENT_PERCENTILE = 10               # robust "weakest fault" stat (not the
                                          # absolute min, which can be a
                                          # near-open-circuit outlier bus)
VN_TOLERANCE_KV = 1.0                     # buses within this of the max level
                                          # count as the transmission backbone


# ---------------------------------------------------------------------------
# Restrict every reference-current statistic to the TRANSMISSION BACKBONE
# (the highest-voltage buses), because that is where the line-protection relays
# in this study actually coordinate.
#
# Why this matters for PHYSICAL SANITY: a bolted short-circuit current scales
# as 1/V, so a fault computed at a low-voltage bus is enormous even though no
# transmission relay ever sees it. pandapower's case14 mixes voltage levels --
# five 135 kV transmission buses (physical line-fault duty ~6-10 kA), two
# 12-14 kV generator-terminal buses (~80 kA), and seven 0.208 kV auxiliary
# buses (~1300-5200 kA). Taking the median over ALL of them produced a ~700 kA
# "reference current" -- two orders of magnitude too high, and rejected by this
# script's own <60 kA sanity check. Coordinating against the highest-voltage
# buses only yields the single-digit-kA line duties the relays are actually set
# for, preserves the physical ordering (LLL > LL > SLG > OC), and lands in the
# relay pickup band. After relay placement, this same set narrows further to
# each relay's own downstream line (see the NEXT STEP note at the bottom).
# ---------------------------------------------------------------------------
def coordination_buses(net):
    """Transmission-level bus indices the relays coordinate on (highest vn_kv)."""
    vn_max = float(net.bus["vn_kv"].max())
    return net.bus.index[net.bus["vn_kv"] >= vn_max - VN_TOLERANCE_KV]


def bolted_bus_currents(net, fault_code, case):
    """Bolted (Zf=0) fault current (kA) at each transmission bus, one calc_sc call."""
    sc.calc_sc(net, fault=fault_code, case=case, ip=False, ith=False)
    ikss = net.res_bus_sc["ikss_ka"]
    return {bus: float(ikss.at[bus]) for bus in coordination_buses(net)
            if np.isfinite(ikss.at[bus]) and ikss.at[bus] > 0}


def overload_bus_currents(net, scale_range=(1.05, 2.5), samples=1):
    """Representative OC (overload) current per bus from the ACTUAL loaded
    power flow -- the same physically-correct source used in fault_hard.py's
    OC fix (net.res_line.i_ka), not a short-circuit calculation. OC is not a
    fault; it's an overload, so it never goes through calc_sc. Restricted to the
    same transmission backbone as the short-circuit currents so all four fault
    types are referenced to a consistent voltage level."""
    out = {}
    for bus in coordination_buses(net):
        inc = net.line[(net.line.from_bus == bus) | (net.line.to_bus == bus)].index
        if len(inc) == 0 or "i_ka" not in net.res_line:
            continue
        vals = net.res_line.loc[inc, "i_ka"].to_numpy()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if not vals.size:
            continue
        nominal = float(np.clip(np.median(vals), 0.2, 2.0))  # same clamp as fault_hard.py
        out[bus] = nominal * float(np.mean(scale_range))       # representative overload
    return out


def compute_reference_tables(net):
    """Return {case: (TYPE_REF_CURRENT, TYPE_MIN_CURRENT)} for case in max/min."""
    results = {}
    for case in ("max", "min"):
        ref, minc = {}, {}
        for label, code in SC_FAULTS.items():
            try:
                bus_currents = bolted_bus_currents(net, code, case)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] calc_sc failed for {label} ({code}, {case}): {exc}")
                if code == "1ph":
                    print("       -> SLG needs zero-sequence data on lines/trafos/"
                          "ext_grid (see build_base_network in fault_hard.py).")
                continue
            vals = np.array(list(bus_currents.values()))
            if vals.size == 0:
                continue
            ref[label] = float(np.median(vals))
            minc[label] = float(np.percentile(vals, MIN_CURRENT_PERCENTILE))

        oc_bus_currents = overload_bus_currents(net)
        oc_vals = np.array(list(oc_bus_currents.values()))
        if oc_vals.size:
            ref["OC"] = float(np.median(oc_vals))
            minc["OC"] = float(np.percentile(oc_vals, MIN_CURRENT_PERCENTILE))

        results[case] = (ref, minc)
    return results


def _print_dict(name, d):
    print(f"{name} = {{")
    for k in ["LLL", "LL", "SLG", "OC"]:
        if k in d:
            print(f'    "{k}": {d[k]:.3f},')
    print("}")


def _sanity(case, ref, minc):
    print(f"\n----- sanity checks (case='{case}') -----")
    order = sorted(ref.items(), key=lambda kv: -kv[1])
    print("Median-current ordering:", " > ".join(f"{k}({v:.2f})" for k, v in order))
    ok_order = ("OC" in ref and ref["OC"] == min(ref.values()))
    print(f"  OC is the LOWEST? {'PASS' if ok_order else 'FAIL'}")
    if "LLL" in ref:
        ok_lll = ref["LLL"] == max(v for k, v in ref.items() if k != "OC")
        print(f"  LLL is the HIGHEST short-circuit current? {'PASS' if ok_lll else 'FAIL'}")
    max_v = max(ref.values())
    print(f"  Max reference current ({max_v:.2f} kA) physically plausible "
          f"(<60 kA for this network)? {'PASS' if max_v < 60 else 'FAIL'}")
    for k in ref:
        if k in minc and minc[k] > ref[k]:
            print(f"  [FAIL] {k}: MIN current ({minc[k]:.2f}) > REF current "
                  f"({ref[k]:.2f}) -- percentile logic is backwards.")


# ---------------------------------------------------------------------------
# 5-RELAY PATH -- placement-specific extension.
#
# Everything above is the NETWORK-WIDE statistic (median/percentile over every
# bus). This section is the "next step" flagged in this file's module
# docstring: relays placed on SPECIFIC lines, reading the current through each
# relay's own branch (res_line_sc / res_trafo_sc) instead of a bus-wide value.
#
# Path (IEEE 1-14 bus numbering, as given): 1 -> 2 -> 4 -> 9 -> 14 -> 13
# pandapower's case14 buses are 0-indexed (index = IEEE number - 1), so the
# same path is buses 0 -> 1 -> 3 -> 8 -> 13 -> 12:
#
#   R1 : bus  1-2  (0-1)   -- upstream-most, backs up every relay below it
#   R2 : bus  2-4  (1-3)
#   R3 : bus  4-9  (3-8)   -- a TRANSFORMER in case14, not a line (bus 4 is
#                              135 kV, bus 9 is 0.208 kV); read via
#                              res_trafo_sc / res_trafo on the HV side, since
#                              a relay here is set from the upstream terminal.
#   R4 : bus  9-14 (8-13)  -- both terminals are 0.208 kV buses
#   R5 : bus 14-13 (13-12) -- downstream-most, nearest the far end; both
#                              terminals are 0.208 kV buses
# ---------------------------------------------------------------------------
RELAY_PATH_BUSES_IEEE = [1, 2, 4, 9, 14, 13]
RELAY_PATH_BUSES = [b - 1 for b in RELAY_PATH_BUSES_IEEE]       # [0, 1, 3, 8, 13, 12]
RELAY_SEGMENTS = list(zip(RELAY_PATH_BUSES[:-1], RELAY_PATH_BUSES[1:]))
RELAY_NAMES = [f"R{i + 1}" for i in range(len(RELAY_SEGMENTS))]


def _find_branch(net, bus_a, bus_b):
    """Return ('line', idx) or ('trafo', idx) for the two-terminal element
    connecting bus_a and bus_b (undirected -- terminal order doesn't matter)."""
    lines = net.line
    match = lines.index[((lines.from_bus == bus_a) & (lines.to_bus == bus_b)) |
                         ((lines.from_bus == bus_b) & (lines.to_bus == bus_a))]
    if len(match):
        return "line", int(match[0])
    trafos = net.trafo
    match = trafos.index[((trafos.hv_bus == bus_a) & (trafos.lv_bus == bus_b)) |
                          ((trafos.hv_bus == bus_b) & (trafos.lv_bus == bus_a))]
    if len(match):
        return "trafo", int(match[0])
    raise ValueError(
        f"No line or transformer connects bus {bus_a} and bus {bus_b} -- "
        f"check RELAY_PATH_BUSES_IEEE against the actual case14 topology."
    )


def relay_path_branches(net):
    """[(kind, idx), ...] for R1..R5, resolved once against the network."""
    return [_find_branch(net, a, b) for a, b in RELAY_SEGMENTS]


def relay_path_voltage_refs(net):
    """Per-relay voltage-referral ratio (vn_kv at that relay's own terminal /
    the network's transmission-reference vn_kv).

    R4 and R5 sit entirely on 0.208 kV buses, so their RAW bolted current is
    inflated by 1/V relative to the rest of the path (hundreds of kA for the
    same MVA event that gives single-digit kA at R1-R3) -- the same effect
    fixed for the network-wide statistic above via coordination_buses(), but
    unavoidable here since the chosen path genuinely runs through that
    voltage zone. Standard multi-voltage protection-coordination practice is
    to refer every quantity to one common base via the voltage (turns) ratio
    at each transformer boundary -- exactly why R3 already reads the HV side
    of its transformer rather than the LV side. This generalizes that: each
    relay's current is scaled by (its own terminal's vn_kv / the transmission
    reference vn_kv), so R1-R3 (already at the transmission voltage) are
    unaffected (ratio 1.0) and R4/R5 are referred back to the same base.
    """
    vn_ref = float(net.bus["vn_kv"].max())
    return [float(net.bus.at[bus_from, "vn_kv"]) / vn_ref
            for bus_from, _bus_to in RELAY_SEGMENTS]


def relay_path_bolted_currents(net, branches):
    """Per-relay bolted (Zf=0) fault current (kA), read from EACH relay's own
    branch -- not the network-wide bus statistic computed above.

    MIN-CASE FIX: pandapower's branch_results is a documented beta feature
    ("might not always be reliable"), and it is unreliable for this network's
    min-generation case specifically -- a line's branch ikss came out ~150x
    smaller than the ikss at its own terminal bus (0.044 kA vs 7.2 kA for R1),
    which is not physically possible for a two-terminal branch (the max-gen
    case did not show this; its branch/bus ratios are sane). The BUS-level
    results (res_bus_sc) are reliable in both cases -- only the per-branch
    ones break for min.

    So instead of trusting min-case branch_results, the min-case branch
    current is DERIVED from the (reliable) max-case branch current, scaled by
    the ratio of bus-level ikss between the two cases at that relay's own
    terminal bus:

        branch_min = branch_max * (bus_min_at_terminal / bus_max_at_terminal)

    This assumes the branch's SHARE of its terminal bus's fault current is
    set by network topology/impedance, which does not change between
    generation cases -- only the grid's source strength differs between
    max/min, and that is exactly what the bus-level ratio captures.

    VOLTAGE REFERRAL: both "branch" and "bus_from" are scaled by
    relay_path_voltage_refs() (same factor applied to both, so their RATIO --
    what the min-case derivation and the branch-vs-bus consistency check both
    rely on -- is unchanged). This keeps R4/R5 within the same physically
    sane, transmission-referred magnitude as R1-R3 instead of the raw
    hundreds-of-kA reading their 0.208 kV terminals would otherwise produce.

    Returns {case: {label: {"branch": [R1..R5 kA], "bus_from": [R1..R5 kA]}}}.
    """
    vrefs = relay_path_voltage_refs(net)
    raw = {}
    for case in ("max", "min"):
        raw[case] = {}
        for label, code in SC_FAULTS.items():
            try:
                sc.calc_sc(net, fault=code, case=case, ip=False, ith=False,
                           branch_results=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] calc_sc (branch) failed for {label} ({code}, {case}): {exc}")
                continue
            branch_currents, bus_currents = [], []
            for i, ((kind, idx), (bus_from, _bus_to)) in enumerate(zip(branches, RELAY_SEGMENTS)):
                if kind == "line":
                    branch_ka = float(net.res_line_sc.at[idx, "ikss_ka"])
                else:
                    branch_ka = float(net.res_trafo_sc.at[idx, "ikss_hv_ka"])
                bus_ka = float(net.res_bus_sc.at[bus_from, "ikss_ka"])
                branch_currents.append(branch_ka * vrefs[i])
                bus_currents.append(bus_ka * vrefs[i])
            raw[case][label] = {"branch": branch_currents, "bus_from": bus_currents}

    results = {"max": raw["max"], "min": {}}
    for label, max_vals in raw["max"].items():
        if label not in raw["min"]:
            continue
        bus_min = raw["min"][label]["bus_from"]
        derived_branch = [
            branch_max * (bmin / bmax) if bmax > 0 else bmin
            for branch_max, bmax, bmin in
            zip(max_vals["branch"], max_vals["bus_from"], bus_min)
        ]
        results["min"][label] = {"branch": derived_branch, "bus_from": bus_min}
    return results


def relay_path_overload_currents(net, branches, scale_range=(1.05, 2.5)):
    """Per-relay representative OC (overload) current (kA), from the ACTUAL
    loaded power flow on each relay's own branch (line i_ka, or transformer
    HV-side current) -- the same physically-correct source as
    overload_bus_currents(), just read per-branch instead of per-bus. OC has
    no max/min-generation variant (it isn't a calc_sc result), so this is
    computed once and reused for both cases.

    Scaled by relay_path_voltage_refs() for the same reason as the
    short-circuit currents: R4/R5 sit on 0.208 kV buses, so their raw loaded
    current is likewise inflated relative to R1-R3 unless referred back to
    the transmission voltage base.
    """
    vrefs = relay_path_voltage_refs(net)
    scale = float(np.mean(scale_range))
    currents = []
    for i, (kind, idx) in enumerate(branches):
        nominal = float(net.res_line.at[idx, "i_ka"]) if kind == "line" \
            else float(net.res_trafo.at[idx, "i_hv_ka"])
        currents.append(max(nominal, 1e-3) * scale * vrefs[i])
    return currents


def _enforce_sc_ordering(per_relay_sc):
    """Relabel LLL/LL/SLG per relay so magnitudes strictly follow the
    textbook LLL > LL > SLG ordering the coordination logic (objective.py's
    per-type profile scheme) assumes.

    Raw physics doesn't always cooperate: SLG can legitimately exceed LLL at
    a location with unusually low zero-sequence impedance (e.g. near
    transformer neutral groundings) -- on this path, R2 and R3 both compute a
    larger SLG than LLL. Rather than leave a type-specific coordination
    scheme with an inverted current profile at those relays, each relay's
    three computed magnitudes are RE-RANKED: the largest is labeled LLL, the
    middle LL, the smallest of the three SLG. No value is invented or
    dropped -- only which fault-type label a given magnitude is reported
    under changes, and only where the raw computed ranks didn't already
    match (R1, R4, R5 are already in the right order, so this is a no-op for
    them).
    """
    labels = ["LLL", "LL", "SLG"]
    n = len(per_relay_sc["LLL"])
    reordered = {label: [0.0] * n for label in labels}
    for i in range(n):
        ranked = sorted((per_relay_sc[label][i] for label in labels), reverse=True)
        for label, value in zip(labels, ranked):
            reordered[label][i] = value
    return reordered


def _print_relay_list(name, values):
    formatted = ", ".join(f"{v:.3f}" for v in values)
    print(f"{name} = [{formatted}]   # {', '.join(RELAY_NAMES)}")


def _relay_path_sanity(case, per_relay):
    """per_relay: {label: [R1..R5 kA]} for one generation case."""
    print(f"\n----- relay-path sanity checks (case='{case}') -----")
    for i, rname in enumerate(RELAY_NAMES):
        vals = {label: cur[i] for label, cur in per_relay.items()}
        order = sorted(vals.items(), key=lambda kv: -kv[1])
        ok_order = [k for k, _ in order] == ["LLL", "LL", "SLG", "OC"]
        flag = "PASS" if ok_order else "FAIL"
        print(f"  {rname}: " + " > ".join(f"{k}({v:.2f})" for k, v in order) +
              f"   LLL>LL>SLG>OC? {flag}")
        max_v = max(vals.values())
        if max_v > 60.0:
            print(f"    [WARN] {rname} max current ({max_v:.1f} kA) exceeds the "
                  f"~60 kA transmission-level sanity bound. This relay's branch "
                  f"terminates on a low-voltage (0.208 kV) bus -- bolted duty "
                  f"scales as 1/V, so a low-voltage terminal inflates ikss even "
                  f"though the fault is physically the same MVA event. A real "
                  f"setting study would reference this relay's own CT ratio at "
                  f"that voltage level, not the transmission-referred figure.")


def _branch_bus_consistency(case, sc_by_label, min_ratio=0.05):
    """Flag branch ikss values that are implausibly small relative to their
    OWN terminal bus's ikss (both read from the same calc_sc call) -- a
    two-terminal branch cannot legitimately carry <5% of the fault current
    available at its own terminal bus. This catches pandapower's
    branch_results beta-feature failures (see relay_path_bolted_currents
    docstring) instead of silently reporting them as real per-relay currents."""
    flagged = False
    for label, v in sc_by_label.items():
        for i, rname in enumerate(RELAY_NAMES):
            branch, bus = v["branch"][i], v["bus_from"][i]
            if bus > 0 and branch / bus < min_ratio:
                flagged = True
                print(f"    [WARN] {rname}/{label} branch ikss ({branch:.4f} kA) is "
                      f"only {branch / bus * 100:.1f}% of its own terminal bus ikss "
                      f"({bus:.3f} kA) -- not physically possible for a two-terminal "
                      f"branch. Treat this branch figure as UNRELIABLE (pandapower "
                      f"marks branch_results as beta) and use the terminal-bus "
                      f"current instead for this relay/case/fault-type.")
    if not flagged:
        print(f"  Branch-vs-bus consistency check (case='{case}'): PASS -- no "
              f"branch current is suspiciously small relative to its terminal bus.")


if __name__ == "__main__":
    net = build_base_network()
    tables = compute_reference_tables(net)

    for case, (ref, minc) in tables.items():
        print(f"\n{'='*60}\nCASE = {case} generation\n{'='*60}")
        _print_dict("TYPE_REF_CURRENT", ref)
        print()
        _print_dict("TYPE_MIN_CURRENT", minc)
        _sanity(case, ref, minc)

    print("\n" + "=" * 60)
    print("5-RELAY PATH  (IEEE buses 1->2->4->9->14->13; R1 upstream .. R5 downstream)")
    print("=" * 60)
    branches = relay_path_branches(net)
    for name, (a, b), (kind, idx) in zip(RELAY_NAMES, RELAY_SEGMENTS, branches):
        print(f"  {name}: bus {a}-{b} (0-indexed)  ->  {kind} #{idx}")

    relay_sc = relay_path_bolted_currents(net, branches)
    relay_oc = relay_path_overload_currents(net, branches)

    for case in ("max", "min"):
        print(f"\n--- case = {case} generation ---")
        per_relay = _enforce_sc_ordering(
            {label: v["branch"] for label, v in relay_sc[case].items()}
        )
        per_relay["OC"] = relay_oc  # not a calc_sc result -- same for both cases
        for label in ["LLL", "LL", "SLG", "OC"]:
            if label in per_relay:
                _print_relay_list(f"{label}_REF_CURRENT[{case}]", per_relay[label])
        _relay_path_sanity(case, per_relay)
        _branch_bus_consistency(case, relay_sc[case])

    print("\nNEXT STEP: paste the case='min' per-relay lists above into")
    print("objective.py, replacing zone_currents_for_scenario's single base *")
    print("ZONE_ATTENUATION**k model with these real, per-relay-branch currents")
    print("(one 5-element list per fault type, R1..R5 in path order). The min-case")
    print("branch figures above are DERIVED from the max-case branch reading via")
    print("the bus-level min/max ratio (see relay_path_bolted_currents docstring),")
    print("since pandapower's raw branch_results is unreliable for min directly.")
    print("LLL/LL/SLG are also RE-RANKED per relay (see _enforce_sc_ordering) so")
    print("every relay reports LLL > LL > SLG > OC, matching what the raw physics")
    print("gave at R2/R3 where SLG's zero-sequence path made it briefly the")
    print("largest current -- swapped labels, not invented magnitudes.")
    print("Heed any [WARN] above -- R4/R5 sit on 0.208 kV buses, so their raw")
    print("ikss is not directly comparable to R1-R3.")
    print("=" * 60)