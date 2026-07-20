"""
fault_option2_hard.py  --  OPTION 2: physically-sane BUT genuinely hard.

Same physically-correct core as Option 1 (magnitudes sane, ordering
LLL >= LL >= SLG > OC in the bulk, sequence features from noisy phases), PLUS a
realistic hard-sensing regime so classification is difficult for HONEST,
physical reasons -- not because the magnitudes are broken:

  1. Heavy CT saturation (frequent, severe) that distorts phase currents and
     corrupts the sequence estimates -- the real reason sequence-based
     discrimination degrades in practice.
  2. A genuine HIGH-IMPEDANCE-FAULT (HIF) population: a fraction of LLL faults
     are drawn at high Zf, where the fault current sags INTO the overload band
     and the retained voltage rises toward normal -- so HIF-LLL genuinely
     overlaps OC in both current and voltage. This is a well-known hard problem
     in protection (HIF detection).
  3. Occasional measurement dropout (a CT/VT channel degraded), forcing the
     model to cope with missing signal.

Accuracy will be BELOW 100% for real reasons -- report whatever it honestly is
(do NOT tune the knobs to hit a target number). It will likely land in the
mid-80s to low-90s depending on seed. The difficulty here is defensible: it
comes from CT saturation and high-impedance faults, both physically real.

Classes: LLL, LL, SLG, OC.  REQUIRES pandapower; run the sanity checks.
"""

from __future__ import annotations
import cmath, math
from pathlib import Path
import numpy as np, pandas as pd
import pandapower as pp
import pandapower.networks as nw
import pandapower.shortcircuit as sc

BASE_MVA = 100.0
C_FACTOR = {"max": 1.10, "min": 1.00}
CT_MAG_ERR, CT_ANG_ERR_DEG = 0.05, 3.0       # heavier sensing noise
VT_MAG_ERR, VT_ANG_ERR_DEG = 0.02, 1.5
LOAD_BACKGROUND_KA = 0.25
CT_FLOOR_KA = 0.05
VT_FLOOR_PU = 0.005
MAX_FAULT_KA = 50.0
Z_MEAN_OHM = 0.25

SAT_FRACTION = 0.60          # fraction of fault samples with CT saturation
HIF_FRACTION = 0.15          # fraction of LLL drawn as high-impedance faults
HIF_Z_MEAN = 12.0            # mean Zf (ohm) for the HIF population
DROPOUT_PROB = 0.05          # per-sample chance one channel is degraded

SAMPLES_PER_CELL = 40
OC_SAMPLES_PER_BUS = 480
RNG = np.random.default_rng(42)

SC_FAULTS = {"LLL": "3ph", "LL": "2ph", "SLG": "1ph"}
_A = cmath.exp(2j * math.pi / 3.0)


def _seq_to_phase(s0, s1, s2):
    return (s0+s1+s2, s0+_A**2*s1+_A*s2, s0+_A*s1+_A**2*s2)

def _phase_to_seq(pa, pb, pc):
    return ((pa+pb+pc)/3.0, (pa+_A*pb+_A**2*pc)/3.0, (pa+_A**2*pb+_A*pc)/3.0)

def _ct_saturate(iph, knee_ka, sev):
    m = abs(iph)/1e3
    if sev <= 0 or m <= knee_ka:
        return iph
    comp = (knee_ka + (m-knee_ka)*(1.0-sev))*(1.0+RNG.normal(0.0, 0.12*sev))
    return cmath.rect(comp*1e3, cmath.phase(iph)+RNG.normal(0.0, 0.35*sev))

def _measure(phasor, mag_err, ang_err_rad, floor_abs=0.0, background_abs=0.0):
    if background_abs > 0.0:
        phasor = phasor + cmath.rect(abs(RNG.normal(0.0, background_abs)),
                                     RNG.uniform(0.0, 2.0*math.pi))
    z = cmath.rect(abs(phasor)*(1.0+RNG.normal(0.0, mag_err)),
                   cmath.phase(phasor)+RNG.normal(0.0, ang_err_rad))
    if floor_abs > 0.0:
        z = z + complex(RNG.normal(0.0, floor_abs), RNG.normal(0.0, floor_abs))
    return z


def build_base_network():
    net = nw.case14()
    net.ext_grid["s_sc_max_mva"] = 1000.0
    net.ext_grid["s_sc_min_mva"] = 800.0
    net.ext_grid["rx_max"] = 0.1; net.ext_grid["rx_min"] = 0.1
    net.ext_grid["x0x_max"] = 1.0; net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_min"] = 1.0; net.ext_grid["r0x0_min"] = 0.1
    if not net.gen.empty:
        cphi = 0.85
        net.gen["vn_kv"] = net.bus.loc[net.gen.bus.values, "vn_kv"].values
        net.gen["sn_mva"] = (net.gen["max_p_mw"].fillna(100.0)/cphi).round(1)
        net.gen["cos_phi"] = cphi; net.gen["xdss_pu"] = 0.20
        net.gen["rdss_ohm"] = (0.02*net.gen["vn_kv"]**2/net.gen["sn_mva"]).round(6)
        net.gen["pg_percent"] = 0.0; net.gen["power_station_trafo"] = np.nan
    if not net.trafo.empty:
        net.trafo["vector_group"] = "Dyn"
        net.trafo["vk0_percent"] = net.trafo["vk_percent"]
        net.trafo["vkr0_percent"] = net.trafo["vkr_percent"]
        net.trafo["mag0_percent"] = 100.0; net.trafo["mag0_rx"] = 0.0
        net.trafo["si0_hv_partial"] = 0.9
    if "r0_ohm_per_km" not in net.line.columns or net.line["r0_ohm_per_km"].isna().any():
        net.line["r0_ohm_per_km"] = net.line["r_ohm_per_km"]*3.0
        net.line["x0_ohm_per_km"] = net.line["x_ohm_per_km"]*3.0
        net.line["c0_nf_per_km"] = net.line["c_nf_per_km"]*0.6
        net.line["endtemp_degree"] = 80.0
    pp.runpp(net, algorithm="nr", init="auto")
    if not net["converged"]:
        raise RuntimeError("Base load flow did not converge.")
    return net


def _sequence_impedances(net, case):
    sc.calc_sc(net, fault="1ph", case=case, ip=False, ith=False)
    res = net.res_bus_sc; z = {}
    for bus in net.bus.index:
        r1, x1 = float(res.at[bus, "rk_ohm"]), float(res.at[bus, "xk_ohm"])
        r0, x0 = float(res.at[bus, "rk0_ohm"]), float(res.at[bus, "xk0_ohm"])
        z[bus] = (complex(r1, x1), complex(r0, x0)) if not math.isnan(r1) else (complex(math.nan), complex(math.nan))
    return z


def _ideal_seq_currents(label, e, z1, z0, zf):
    z2 = z1
    if label == "LLL": return 0j, e/(z1+zf), 0j
    if label == "LL":  i1 = e/(z1+z2+zf); return 0j, i1, -i1
    if label == "SLG": i = e/(z1+z2+z0+3.0*zf); return i, i, i
    raise ValueError(label)


def _bus_loading(net, bus):
    return float(net.load.loc[net.load.bus == bus, "p_mw"].sum())/BASE_MVA


def _apply_dropout(ia, ib, ic):
    """Occasionally degrade one current channel (stuck low)."""
    if RNG.random() < DROPOUT_PROB:
        ch = RNG.integers(0, 3)
        factor = RNG.uniform(0.0, 0.3)
        if ch == 0: ia = ia*factor
        elif ch == 1: ib = ib*factor
        else: ic = ic*factor
    return ia, ib, ic


def generate_sc_rows(net):
    rows = []; vpre = net.res_bus.vm_pu.copy()
    ct_ang = math.radians(CT_ANG_ERR_DEG); vt_ang = math.radians(VT_ANG_ERR_DEG)
    i_bg, i_fl = LOAD_BACKGROUND_KA*1e3, CT_FLOOR_KA*1e3
    for case in ("max", "min"):
        try:
            zseq = _sequence_impedances(net, case)
        except Exception as exc:
            print(f"[WARN] calc_sc(1ph) failed ({case}): {exc}"); continue
        c = C_FACTOR[case]
        for label in SC_FAULTS:
            for bus in net.bus.index:
                z1, z0 = zseq[bus]
                if cmath.isnan(z1): continue
                vn_v = net.bus.at[bus, "vn_kv"]*1000.0
                e = c*vn_v/math.sqrt(3.0); vn_ph = vn_v/math.sqrt(3.0)
                ibase = BASE_MVA/(math.sqrt(3.0)*net.bus.at[bus, "vn_kv"])
                v_fl = VT_FLOOR_PU*vn_ph; knee = RNG.uniform(4.0, 10.0)
                for _ in range(SAMPLES_PER_CELL):
                    # HIF population: some LLL drawn at high impedance (sags into OC band)
                    if label == "LLL" and RNG.random() < HIF_FRACTION:
                        zf = min(RNG.exponential(HIF_Z_MEAN), 40.0)
                    else:
                        zf = min(RNG.exponential(Z_MEAN_OHM), 8.0)
                    i0, i1, i2 = _ideal_seq_currents(label, e, z1, z0, zf)
                    v1, v2, v0 = e-i1*z1, -i2*z1, -i0*z0
                    ia, ib, ic = _seq_to_phase(i0, i1, i2)
                    va, vb, vc = _seq_to_phase(v0, v1, v2)
                    if max(abs(ia), abs(ib), abs(ic))/1e3 > MAX_FAULT_KA:
                        continue
                    # heavy CT saturation (per-phase asymmetric)
                    if RNG.random() < SAT_FRACTION:
                        base_sev = RNG.uniform(0.2, 0.75)
                        ia = _ct_saturate(ia, knee, base_sev*RNG.uniform(0.6, 1.0))
                        ib = _ct_saturate(ib, knee, base_sev*RNG.uniform(0.6, 1.0))
                        ic = _ct_saturate(ic, knee, base_sev*RNG.uniform(0.6, 1.0))
                    ia, ib, ic = _apply_dropout(ia, ib, ic)
                    ia = _measure(ia, CT_MAG_ERR, ct_ang, i_fl, i_bg)
                    ib = _measure(ib, CT_MAG_ERR, ct_ang, i_fl, i_bg)
                    ic = _measure(ic, CT_MAG_ERR, ct_ang, i_fl, i_bg)
                    va = _measure(va, VT_MAG_ERR, vt_ang, v_fl)
                    vb = _measure(vb, VT_MAG_ERR, vt_ang, v_fl)
                    vc = _measure(vc, VT_MAG_ERR, vt_ang, v_fl)
                    s0, s1, s2 = _phase_to_seq(ia, ib, ic)
                    rows.append(_row(label, bus, zf, vpre.at[bus], ia, ib, ic,
                                     va, vb, vc, s0, s1, s2, ibase, vn_ph,
                                     _bus_loading(net, bus)))
    return rows


def generate_oc_rows(net):
    rows = []; ct_ang = math.radians(CT_ANG_ERR_DEG); vt_ang = math.radians(VT_ANG_ERR_DEG)
    i_bg, i_fl = LOAD_BACKGROUND_KA*1e3, CT_FLOOR_KA*1e3
    for bus in net.bus.index:
        vn_v = net.bus.at[bus, "vn_kv"]*1000.0; vn_ph = vn_v/math.sqrt(3.0)
        ibase = BASE_MVA/(math.sqrt(3.0)*net.bus.at[bus, "vn_kv"]); v_fl = VT_FLOOR_PU*vn_ph
        # Physical NOMINAL line current at this bus from the actual power flow
        # (res_line.i_ka), NOT net.line.max_i_ka -- case14's max_i_ka is
        # unpopulated/garbage (returns ~28000 for most lines), which was
        # producing 100,000 kA "overloads". res_line.i_ka is the real loaded
        # current in kA. Fall back to a sane 0.5 kA if unavailable.
        inc = net.line[(net.line.from_bus == bus) | (net.line.to_bus == bus)].index
        nominal_ka = 0.5
        if len(inc) and "i_ka" in net.res_line:
            vals = net.res_line.loc[inc, "i_ka"].to_numpy()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                nominal_ka = float(np.median(vals))
        # clamp nominal to a physically sane band so no bus can explode
        nominal_ka = float(np.clip(nominal_ka, 0.2, 2.0))
        knee = RNG.uniform(4.0, 10.0)
        for _ in range(OC_SAMPLES_PER_BUS):
            # overload = 1.05-3x the nominal loaded current (an overload, by
            # definition, stays well BELOW short-circuit levels)
            scale = RNG.uniform(1.05, 3.2)
            base = nominal_ka*scale*1e3          # kA -> amps (phase convention)
            unb = 1.0 + RNG.normal(0.0, 0.05, 3)
            ia, ib, ic = _seq_to_phase(0j, complex(base), 0j)
            ia, ib, ic = ia*unb[0], ib*unb[1], ic*unb[2]
            if RNG.random() < 0.35:
                sev = RNG.uniform(0.1, 0.4)
                ia = _ct_saturate(ia, knee, sev); ib = _ct_saturate(ib, knee, sev); ic = _ct_saturate(ic, knee, sev)
            ia, ib, ic = _apply_dropout(ia, ib, ic)
            ia = _measure(ia, CT_MAG_ERR, ct_ang, i_fl, i_bg)
            ib = _measure(ib, CT_MAG_ERR, ct_ang, i_fl, i_bg)
            ic = _measure(ic, CT_MAG_ERR, ct_ang, i_fl, i_bg)
            sag = RNG.uniform(0.80, 0.97)
            va, vb, vc = [_measure(complex(sag*vn_ph), VT_MAG_ERR, vt_ang, v_fl) for _ in range(3)]
            s0, s1, s2 = _phase_to_seq(ia, ib, ic)
            rows.append(_row("OC", bus, 0.0, sag, ia, ib, ic, va, vb, vc,
                             s0, s1, s2, ibase, vn_ph, scale))
    return rows


def _row(label, bus, zf, vpre, ia, ib, ic, va, vb, vc, s0, s1, s2, ibase, vn_ph, loading):
    ia_ka, ib_ka, ic_ka = abs(ia)/1e3, abs(ib)/1e3, abs(ic)/1e3
    i0_ka, i1_ka, i2_ka = abs(s0)/1e3, abs(s1)/1e3, abs(s2)/1e3
    va_pu, vb_pu, vc_pu = abs(va)/vn_ph, abs(vb)/vn_ph, abs(vc)/vn_ph
    return {
        "fault_type": label, "fault_bus": int(bus),
        "fault_impedance_ohm": round(float(zf), 4),
        "pre_fault_voltage_pu": round(float(vpre), 4),
        "fault_voltage_pu": round(float(np.clip(min(va_pu, vb_pu, vc_pu), 0.0, 1.10)), 4),
        "fault_current_pu": round(max(ia_ka, ib_ka, ic_ka)/ibase, 4),
        "fault_current_ka": round(max(ia_ka, ib_ka, ic_ka), 4),
        "Ia_ka": round(ia_ka, 4), "Ib_ka": round(ib_ka, 4), "Ic_ka": round(ic_ka, 4),
        "I1_ka": round(i1_ka, 4), "I2_ka": round(i2_ka, 4), "I0_ka": round(i0_ka, 4),
        "Va_pu": round(va_pu, 4), "Vb_pu": round(vb_pu, 4), "Vc_pu": round(vc_pu, 4),
        "loading_pu": round(float(loading), 4),
    }


def generate_fault_dataset(output_csv="fault_dataset_hard.csv"):
    net = build_base_network()
    rows = generate_sc_rows(net) + generate_oc_rows(net)
    df = pd.DataFrame(rows); df.insert(0, "row_id", range(1, len(df)+1))
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df):,} rows to {Path(output_csv).resolve()}")
    print(df.groupby("fault_type").size().to_string())
    _sanity(df)
    return df


def _sanity(df):
    print("\n===== SANITY (Option 2: hard) =====")
    med = df.groupby("fault_type")["fault_current_ka"].median().sort_values(ascending=False)
    print("Median current ordering (LLL should lead; OC must be LOWEST):")
    print(med.round(3).to_string())
    mx = df.groupby("fault_type")["fault_current_ka"].max()
    print(f"\nMax current per class (kA), want all < {MAX_FAULT_KA}:")
    print(mx.round(2).to_string())

    # Hard assertions that would have caught the OC blow-up:
    oc_med = df[df.fault_type == "OC"]["fault_current_ka"].median()
    sc_med_min = df[df.fault_type.isin(["LLL", "LL", "SLG"])] \
        .groupby("fault_type")["fault_current_ka"].median().min()
    print("\nChecks:")
    print(f"  OC median ({oc_med:.2f} kA) < smallest SC-class median "
          f"({sc_med_min:.2f} kA)?  {'PASS' if oc_med < sc_med_min else 'FAIL <-- OC too high'}")
    print(f"  OC max ({df[df.fault_type=='OC'].fault_current_ka.max():.2f} kA) "
          f"< {MAX_FAULT_KA} kA?  {'PASS' if df[df.fault_type=='OC'].fault_current_ka.max() < MAX_FAULT_KA else 'FAIL'}")
    print(f"  whole-dataset max ({df.fault_current_ka.max():.2f} kA) < {MAX_FAULT_KA}?  "
          f"{'PASS' if df.fault_current_ka.max() < MAX_FAULT_KA else 'FAIL'}")
    print("\nDifficulty comes from CT saturation + high-impedance faults + dropout "
          "(all physical). Report the accuracy you actually get -- do NOT tune "
          "the knobs to hit a target number.")


if __name__ == "__main__":
    generate_fault_dataset()