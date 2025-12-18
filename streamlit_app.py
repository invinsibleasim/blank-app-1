
# -*- coding: utf-8 -*-
# Streamlit app: PV Module IV Measurement Uncertainty Calculator (extended)
# File: pv_iv_uncertainty_app.py

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

DIST_RECTANGULAR = "Rectangular"
DIST_NORMAL = "Normal"

def standard_uncertainty(value: float, distribution: str) -> float:
    # Return standard uncertainty (1-sigma). Rectangular: a/sqrt(3); Normal: sigma.
    if distribution == DIST_RECTANGULAR:
        return float(value) / math.sqrt(3)
    return float(value)

def combine_rss(values: List[float]) -> float:
    # Root-sum-of-squares
    arr = np.array(values, dtype=float)
    return float(np.sqrt(np.sum(arr ** 2)))

def u_temp_effect(tc_percent_per_k: float, u_tc_percent_per_k: float, dT_k: float, u_dT_k: float) -> float:
    # Uncertainty of y = TC[%/K] * dT[K]: u(y)^2 = (dT)^2*u(TC)^2 + (TC)^2*u(dT)^2.
    return float(np.sqrt((dT_k ** 2) * (u_tc_percent_per_k ** 2) + (tc_percent_per_k ** 2) * (u_dT_k ** 2)))

def percent_from_temp_dev_c(dev_c: float) -> float:
    # Convert temperature deviation (degC) to relative percent via (dev/25)*100.
    return float(dev_c) / 25.0 * 100.0

st.set_page_config(page_title="PV IV Measurement Uncertainty (Extended)", layout="wide")

st.title("PV Module IV Measurement Uncertainty Calculator – Extended")
st.caption("GUM-based uncertainty budgets for Isc, Impp, Voc, Vmpp, Pmpp with extended contributors.")

with st.sidebar:
    st.header("Global settings")
    k = st.number_input("Coverage factor k (approx 2 for ~95%)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    dist_default = st.selectbox("Default distribution for additional % inputs", [DIST_RECTANGULAR, DIST_NORMAL], index=0)
    st.divider()

    st.subheader("Environmental & test configuration")
    t_module = st.number_input("Measured module temperature [degC]", value=25.0, step=0.1)
    t_ref = st.number_input("Reference temperature [degC] (STC=25)", value=25.0, step=0.1)
    dT = t_module - t_ref
    st.write(f"Delta T = {dT:.2f} degC")

    mode = st.selectbox("Measurement mode", ["Default", "Long storage (low handling)", "Fast/normal"], index=0)
    suggested_dev = 1.0 if mode in ("Default", "Fast/normal") else 0.2
    max_temp_dev = st.number_input("Max temperature deviation [degC] (half-width a)", value=float(suggested_dev), step=0.1)
    u_dT = standard_uncertainty(max_temp_dev, DIST_RECTANGULAR)
    st.write(f"Standard uncertainty u(Delta T) = {u_dT:.3f} degC")

    st.markdown("**Geometry / test area (context)**")
    min_len_mm = st.number_input("Min. module length [mm]", value=1000, step=10)
    min_w_mm = st.number_input("Min. module width [mm]", value=600, step=10)
    test_len_m = st.number_input("Test area length [m]", value=2.6, step=0.1)
    test_w_m = st.number_input("Test area width [m]", value=1.3, step=0.1)
    module_area_m2 = (min_len_mm/1000.0) * (min_w_mm/1000.0)
    test_area_m2 = test_len_m * test_w_m
    st.write(f"Module area (min) ~= {module_area_m2:.3f} m^2; Test area ~= {test_area_m2:.3f} m^2")

    # Base contributors
    st.subheader("Irradiance & spectral (base)")
    irr_nonuni = st.number_input("Max irradiance non-uniformity [%] (half-width a)", value=1.6, step=0.1)
    irr_dev_ref = st.number_input("Irradiance deviation module->reference cell [%] (half-width a)", value=0.50, step=0.01)
    mm_sigma = st.number_input("Spectral mismatch (MM) uncertainty [%] (sigma)", value=1.00, step=0.01)

    st.subheader("PV module temperature coefficients (PV under test)")
    tc_isc_pv = st.number_input("TempCo Isc [%/K] (PV module)", value=0.04, step=0.01)
    tc_isc_pv_dev = st.number_input("Uncertainty of TempCo Isc [%/K] (half-width a)", value=0.02, step=0.01)
    u_tc_isc_pv = standard_uncertainty(tc_isc_pv_dev, DIST_RECTANGULAR)

    tc_voc_pv = st.number_input("TempCo Voc [%/K] (PV module)", value=0.40, step=0.01)
    tc_voc_pv_dev = st.number_input("Uncertainty of TempCo Voc [%/K] (half-width a)", value=0.10, step=0.01)
    u_tc_voc_pv = standard_uncertainty(tc_voc_pv_dev, DIST_RECTANGULAR)

    tc_pmax_pv = st.number_input("TempCo Pmax [%/K] (PV module)", value=0.00, step=0.01)
    tc_pmax_pv_dev = st.number_input("Uncertainty of TempCo Pmax [%/K] (half-width a)", value=0.00, step=0.01)
    u_tc_pmax_pv = standard_uncertainty(tc_pmax_pv_dev, DIST_RECTANGULAR)

    st.subheader("Temperature measurement chain (PV under test)")
    temp_dev_backsheet = st.number_input("Backsheet<->cell temperature difference [degC] (half-width a)", value=0.40, step=0.05)
    temp_sensors_mean_dev = st.number_input("Sensors mean<->true mean [degC] (half-width a)", value=0.40, step=0.05)
    include_temp_chain = st.checkbox("Include temp chain contributions in current-related parameters", value=True)

    st.subheader("Include irradiance effects in voltage parameters?")
    include_irr_in_voltage = st.checkbox("Include irradiance-related uncertainties in Voc/Vmpp", value=False)

    st.subheader("Hysteresis (%), Rectangular")
    hyst_isc = st.number_input("Hysteresis Isc [%] (half-width a)", value=0.19, step=0.01)
    hyst_impp = st.number_input("Hysteresis Impp [%] (half-width a)", value=0.10, step=0.01)
    hyst_voc = st.number_input("Hysteresis Voc [%] (half-width a)", value=0.04, step=0.01)
    hyst_vmpp = st.number_input("Hysteresis Vmpp [%] (half-width a)", value=0.15, step=0.01)
    hyst_pmpp = st.number_input("Hysteresis Pmpp [%] (half-width a)", value=0.50, step=0.01)

    st.subheader("Reproducibility (%), Normal (sigma)")
    repro_isc = st.number_input("Reproducibility Isc [%] (sigma)", value=0.65, step=0.01)
    repro_impp = st.number_input("Reproducibility Impp [%] (sigma)", value=0.66, step=0.01)
    repro_voc = st.number_input("Reproducibility Voc [%] (sigma)", value=0.19, step=0.01)
    repro_vmpp = st.number_input("Reproducibility Vmpp [%] (sigma)", value=0.25, step=0.01)
    repro_pmpp = st.number_input("Reproducibility Pmpp [%] (sigma)", value=0.72, step=0.01)

# Extended contributors
st.markdown("## Extended contributors")

with st.expander("Reference module / DAQ / room"):
    daq_refcell = st.number_input("Uncertainty of DAQ Reference Cell or Irradiance Channel [%]", value=0.0, step=0.01)
    temp_room = st.number_input("Temperature in measurement room [degC] range (half-width a)", value=0.0, step=0.1)
    temp_range_target = st.number_input("Temperature range around target temperature [degC] (half-width a)", value=0.0, step=0.1)
    calib_ref_mod = st.number_input("Calibration uncertainty of reference module [%]", value=0.0, step=0.01)
    drift_ref_mod = st.number_input("Drift of reference module [%]", value=0.0, step=0.01)

    tc_isc_ref = st.number_input("Temperature Coefficient for Isc of Reference module [%/K]", value=0.0, step=0.01)
    u_tc_isc_ref = st.number_input("Uncertainty of TempCo for Isc of Reference module [%/K]", value=0.0, step=0.01)
    tc_pmax_ref = st.number_input("Temperature Coefficient for Pmax of Reference module [%/K]", value=0.0, step=0.01)
    u_tc_pmax_ref = st.number_input("Uncertainty of TempCo for Pmax of Reference module [%/K]", value=0.0, step=0.01)

    u_temp_unit_ref = st.number_input("Uncertainty of temperature measurement unit (reference module) [degC] (half-width a)", value=0.0, step=0.05)

with st.expander("Geometry / orientation / irradiance field"):
    dist_refcell_center = st.number_input("Distance of reference cell from center line of module [mm]", value=0.0, step=1.0)
    d1_nom = st.number_input("Nominal distance from flash lamp to module d1 [mm]", value=0.0, step=1.0)
    u_d1 = st.number_input("Uncertainty of distance d1 [mm] (half-width a)", value=0.0, step=0.1)
    u_d2 = st.number_input("Uncertainty of orientation described by d2 [deg or mm eq.]", value=0.0, step=0.1)
    u_avg_irr_area = st.number_input("Uncertainty of average irradiation in measurement area [%]", value=0.0, step=0.01)
    spectral_mismatch_extra = st.number_input("Spectral Mismatch (additional) [%]", value=0.0, step=0.01)
    non_uniformity_module = st.number_input("Non-uniformity of irradiance across PV module [%] (half-width a)", value=0.0, step=0.01)

with st.expander("PV module measurement chain / IV system"):
    u_temp_unit_pv = st.number_input("Uncertainty of temperature measurement unit (PV module) [degC] (half-width a)", value=0.0, step=0.05)
    temp_cell_vs_backsheet = st.number_input("Temperature difference between cell and backsheet [degC] (half-width a)", value=0.0, step=0.05)
    u_ivcurve_eload = st.number_input("Uncertainty of Electronic load IV curve [%]", value=0.0, step=0.01)
    lead_resistance = st.number_input("Resistance between module connectors and 4-wire point [mOhm]", value=0.0, step=0.01)
    max_imp = st.number_input("Maximum current at Imp [A] (info)", value=0.0, step=0.1)
    max_vmp = st.number_input("Maximum voltage at Vmp [V] (info)", value=0.0, step=0.1)
    hyst_didt = st.number_input("Maximum hysteresis or dV/dt & dI/dt during IV measurement [%] (half-width a)", value=0.0, step=0.01)

with st.expander("Stability / reproducibility"):
    very_short = st.number_input("Very short-term stability [%]", value=0.0, step=0.01)
    short_term = st.number_input("Short-term stability [%]", value=0.0, step=0.01)
    long_term = st.number_input("Long-term stability [%]", value=0.0, step=0.01)
    long_rep_pmax = st.number_input("Long-term reproducibility: Pmax [%]", value=0.0, step=0.01)
    long_rep_isc = st.number_input("Long-term reproducibility: Isc [%]", value=0.0, step=0.01)
    long_rep_voc = st.number_input("Long-term reproducibility: Voc [%]", value=0.0, step=0.01)

with st.expander("Methods: STC corrections & IV fitting"):
    stc_pmax = st.number_input("Uncertainty from method for STC correction: Pmax [%]", value=0.0, step=0.01)
    stc_voc = st.number_input("Uncertainty from method for STC correction: Voc [%]", value=0.0, step=0.01)
    stc_isc = st.number_input("Uncertainty from method for STC correction: Isc [%]", value=0.0, step=0.01)

    fit_pmax = st.number_input("Uncertainty from IV parameters fitting procedure: Pmax [%]", value=0.0, step=0.01)
    fit_isc = st.number_input("Uncertainty from IV parameters fitting procedure: Isc [%]", value=0.0, step=0.01)
    fit_voc = st.number_input("Uncertainty from IV parameters fitting procedure: Voc [%]", value=0.0, step=0.01)

# Derived uncertainties (base)
u_mm = standard_uncertainty(mm_sigma, DIST_NORMAL)
u_irr_nonuni = standard_uncertainty(irr_nonuni, DIST_RECTANGULAR)
u_irr_dev_ref = standard_uncertainty(irr_dev_ref, DIST_RECTANGULAR)

temp_chain_percent_1 = percent_from_temp_dev_c(temp_dev_backsheet)
temp_chain_percent_2 = percent_from_temp_dev_c(temp_sensors_mean_dev)
u_temp_chain_1 = standard_uncertainty(temp_chain_percent_1, DIST_RECTANGULAR)
u_temp_chain_2 = standard_uncertainty(temp_chain_percent_2, DIST_RECTANGULAR)

u_temp_I_pv = u_temp_effect(tc_isc_pv, u_tc_isc_pv, dT, u_dT)
u_temp_V_pv = u_temp_effect(tc_voc_pv, u_tc_voc_pv, dT, u_dT)
u_temp_P_pv = u_temp_effect(tc_pmax_pv, u_tc_pmax_pv, dT, u_dT)

u_hyst_isc = standard_uncertainty(hyst_isc, DIST_RECTANGULAR)
u_hyst_impp = standard_uncertainty(hyst_impp, DIST_RECTANGULAR)
u_hyst_voc = standard_uncertainty(hyst_voc, DIST_RECTANGULAR)
u_hyst_vmpp = standard_uncertainty(hyst_vmpp, DIST_RECTANGULAR)
u_hyst_pmpp = standard_uncertainty(hyst_pmpp, DIST_RECTANGULAR)

u_repro_isc = standard_uncertainty(repro_isc, DIST_NORMAL)
u_repro_impp = standard_uncertainty(repro_impp, DIST_NORMAL)
u_repro_voc = standard_uncertainty(repro_voc, DIST_NORMAL)
u_repro_vmpp = standard_uncertainty(repro_vmpp, DIST_NORMAL)
u_repro_pmpp = standard_uncertainty(repro_pmpp, DIST_NORMAL)

# Derived uncertainties (extended)
u_daq_refcell = standard_uncertainty(daq_refcell, dist_default)

u_temp_room = standard_uncertainty(percent_from_temp_dev_c(temp_room), DIST_RECTANGULAR)
u_temp_range_target = standard_uncertainty(percent_from_temp_dev_c(temp_range_target), DIST_RECTANGULAR)

u_calib_ref_mod = standard_uncertainty(calib_ref_mod, dist_default)
u_drift_ref_mod = standard_uncertainty(drift_ref_mod, dist_default)

u_temp_I_ref = u_temp_effect(tc_isc_ref, u_tc_isc_ref, dT, u_dT) if (tc_isc_ref or u_tc_isc_ref) else 0.0
u_temp_P_ref = u_temp_effect(tc_pmax_ref, u_tc_pmax_ref, dT, u_dT) if (tc_pmax_ref or u_tc_pmax_ref) else 0.0

u_temp_unit_ref_percent = standard_uncertainty(percent_from_temp_dev_c(u_temp_unit_ref), DIST_RECTANGULAR)

u_avg_irr_area_std = standard_uncertainty(u_avg_irr_area, dist_default)
nu_spectral_mismatch_extra = standard_uncertainty(spectral_mismatch_extra, dist_default)
nu_nonuniform_module = standard_uncertainty(non_uniformity_module, DIST_RECTANGULAR)

u_temp_unit_pv_percent = standard_uncertainty(percent_from_temp_dev_c(u_temp_unit_pv), DIST_RECTANGULAR)
u_cell_vs_backsheet_percent = standard_uncertainty(percent_from_temp_dev_c(temp_cell_vs_backsheet), DIST_RECTANGULAR)

u_ivcurve_eload_std = standard_uncertainty(u_ivcurve_eload, dist_default)
lead_resistance_percent = 0.0

u_hyst_didt_std = standard_uncertainty(hyst_didt, DIST_RECTANGULAR)

u_very_short = standard_uncertainty(very_short, dist_default)
nu_short_term = standard_uncertainty(short_term, dist_default)
nu_long_term = standard_uncertainty(long_term, dist_default)

u_long_rep_pmax = standard_uncertainty(long_rep_pmax, DIST_NORMAL)
nu_long_rep_isc = standard_uncertainty(long_rep_isc, DIST_NORMAL)
nu_long_rep_voc = standard_uncertainty(long_rep_voc, DIST_NORMAL)

u_stc_pmax = standard_uncertainty(stc_pmax, dist_default)
nu_stc_voc = standard_uncertainty(stc_voc, dist_default)
nu_stc_isc = standard_uncertainty(stc_isc, dist_default)

u_fit_pmax = standard_uncertainty(fit_pmax, dist_default)
nu_fit_isc = standard_uncertainty(fit_isc, dist_default)
nu_fit_voc = standard_uncertainty(fit_voc, dist_default)

# Build budgets
contrib_I_common = {
    "Spectral mismatch (MM)": u_mm,
    "Irradiance non-uniformity (base)": u_irr_nonuni,
    "Irradiance dev. module->ref cell (base)": u_irr_dev_ref,
    "Average irradiation in area": u_avg_irr_area_std,
    "Spectral mismatch (extra)": u_spectral_mismatch_extra,
    "DAQ reference cell / irradiance channel": u_daq_refcell,
    "Calibration ref module": u_calib_ref_mod,
    "Drift ref module": u_drift_ref_mod,
}

if include_temp_chain:
    contrib_I_common.update({
        "Temp chain: backsheet<->cell (PV)": u_cell_vs_backsheet_percent,
        "Temp chain: sensors mean<->true mean (PV)": u_temp_chain_2,
        "Temp unit (PV)": u_temp_unit_pv_percent,
        "Temp unit (ref module)": u_temp_unit_ref_percent,
        "Room temp range around target": u_temp_range_target,
    })

contrib_I_isc_specific = {
    "Temperature effect via PV TC_Isc": u_temp_I_pv,
    "Temperature effect via REF TC_Isc": u_temp_I_ref,
    "Hysteresis": u_hyst_isc,
    "IV curve (electronic load)": u_ivcurve_eload_std,
    "Very short-term stability": u_very_short,
    "Short-term stability": u_short_term,
    "Long-term stability": u_long_term,
    "STC method Isc": u_stc_isc,
    "IV fitting Isc": u_fit_isc,
    "Reproducibility (short-term)": u_repro_isc,
    "Reproducibility (long-term)": u_long_rep_isc,
}

contrib_I_impp_specific = {
    "Temperature effect via PV TC_Isc": u_temp_I_pv,
    "Hysteresis": u_hyst_impp,
    "IV curve (electronic load)": u_ivcurve_eload_std,
    "Very short-term stability": u_very_short,
    "Short-term stability": u_short_term,
    "Long-term stability": u_long_term,
    "STC method Pmax": u_stc_pmax,
    "IV fitting Pmax": u_fit_pmax,
    "Reproducibility (short-term)": u_repro_impp,
}

contrib_V_common = {
    "Temperature effect via PV TC_Voc": u_temp_V_pv,
    "IV curve (electronic load)": u_ivcurve_eload_std,
}

if include_irr_in_voltage:
    contrib_V_common.update({
        "Irradiance non-uniformity (base)": u_irr_nonuni,
        "Irradiance dev. module->ref cell": u_irr_dev_ref,
        "Spectral mismatch (MM)": u_mm,
    })

contrib_V_voc_specific = {
    "Hysteresis": u_hyst_voc,
    "Very short-term stability": u_very_short,
    "Short-term stability": u_short_term,
    "Long-term stability": u_long_term,
    "STC method Voc": u_stc_voc,
    "IV fitting Voc": u_fit_voc,
    "Reproducibility (short-term)": u_repro_voc,
    "Reproducibility (long-term)": u_long_rep_voc,
}

contrib_V_vmpp_specific = {
    "Hysteresis": u_hyst_vmpp,
    "Very short-term stability": u_very_short,
    "Short-term stability": u_short_term,
}

contrib_P_pmpp_specific = {
    "Temperature effect via PV TC_Pmax": u_temp_P_pv,
    "Temperature effect via REF TC_Pmax": u_temp_P_ref,
    "Hysteresis": u_hyst_pmpp,
    "IV curve (electronic load)": u_ivcurve_eload_std,
    "Very short-term stability": u_very_short,
    "Short-term stability": u_short_term,
    "Long-term stability": u_long_term,
    "STC method Pmax": u_stc_pmax,
    "IV fitting Pmax": u_fit_pmax,
    "Reproducibility (short-term)": u_repro_pmpp,
    "Reproducibility (long-term)": u_long_rep_pmax,
}

# Compute

def make_budget_df(contribs: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame({"Contribution": list(contribs.keys()), "Standard uncertainty u_j [%]": list(contribs.values())})
    df["u_j^2"] = df["Standard uncertainty u_j [%]"] ** 2
    return df

contribs_isc = {**contrib_I_common, **contrib_I_isc_specific}
isc_df = make_budget_df(contribs_isc)
uc_isc = combine_rss(list(contribs_isc.values()))
U_isc = k * uc_isc

contribs_impp = {**contrib_I_common, **contrib_I_impp_specific}
impp_df = make_budget_df(contribs_impp)
uc_impp = combine_rss(list(contribs_impp.values()))
U_impp = k * uc_impp

contribs_voc = {**contrib_V_common, **contrib_V_voc_specific}
voc_df = make_budget_df(contribs_voc)
uc_voc = combine_rss(list(contribs_voc.values()))
U_voc = k * uc_voc

contribs_vmpp = {**contrib_V_common, **contrib_V_vmpp_specific}
vmpp_df = make_budget_df(contribs_vmpp)
uc_vmpp = combine_rss(list(contribs_vmpp.values()))
U_vmpp = k * uc_vmpp

uc_pmpp_from_iv = np.sqrt(uc_impp ** 2 + uc_vmpp ** 2)
contribs_pmpp = {"From Impp & Vmpp": uc_pmpp_from_iv, **contrib_P_pmpp_specific}
pmpp_df = make_budget_df(contribs_pmpp)
uc_pmpp = combine_rss(list(contribs_pmpp.values()))
U_pmpp = k * uc_pmpp

# Display
st.subheader("Uncertainty budgets (standard uncertainties in %)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Isc")
    st.dataframe(isc_df, use_container_width=True)
    st.write(f"uc(Isc) = **{uc_isc:.3f}%**; Expanded U (k={k}) = **{U_isc:.3f}%**")

    st.markdown("### Voc")
    st.dataframe(voc_df, use_container_width=True)
    st.write(f"uc(Voc) = **{uc_voc:.3f}%**; Expanded U (k={k}) = **{U_voc:.3f}%**")

with col2:
    st.markdown("### Impp")
    st.dataframe(impp_df, use_container_width=True)
    st.write(f"uc(Impp) = **{uc_impp:.3f}%**; Expanded U (k={k}) = **{U_impp:.3f}%**")

    st.markdown("### Vmpp")
    st.dataframe(vmpp_df, use_container_width=True)
    st.write(f"uc(Vmpp) = **{uc_vmpp:.3f}%**; Expanded U (k={k}) = **{U_vmpp:.3f}%**")

st.markdown("### Pmpp")
st.dataframe(pmpp_df, use_container_width=True)
st.write(f"uc(Pmpp) = **{uc_pmpp:.3f}%**; Expanded U (k={k}) = **{U_pmpp:.3f}%**")

st.subheader("Contribution breakdown")
tabs = st.tabs(["Isc", "Impp", "Voc", "Vmpp", "Pmpp"])
for tab, df in zip(tabs, [isc_df, impp_df, voc_df, vmpp_df, pmpp_df]):
    with tab:
        plot_df = df.sort_values("Standard uncertainty u_j [%]", ascending=True)
        st.bar_chart(plot_df.set_index("Contribution")[['Standard uncertainty u_j [%]']])

st.subheader("Export uncertainty report")
all_tables = {"Isc": isc_df, "Impp": impp_df, "Voc": voc_df, "Vmpp": vmpp_df, "Pmpp": pmpp_df}
summary = pd.DataFrame({
    "Parameter": ["Isc", "Impp", "Voc", "Vmpp", "Pmpp"],
    "uc [%]": [uc_isc, uc_impp, uc_voc, uc_vmpp, uc_pmpp],
    "k": [k] * 5,
    "U = k*uc [%]": [U_isc, U_impp, U_voc, U_vmpp, U_pmpp],
})
with st.expander("Preview summary table"):
    st.dataframe(summary, use_container_width=True)

csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button(label="Download summary (CSV)", data=csv_bytes, file_name="pv_iv_uncertainty_summary.csv", mime="text/csv")

with pd.ExcelWriter("pv_iv_uncertainty_report.xlsx", engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    for key, df in all_tables.items():
        df.to_excel(writer, sheet_name=f"Budget {key}", index=False)

with open("pv_iv_uncertainty_report.xlsx", "rb") as f:
    st.download_button(label="Download full report (Excel)", data=f, file_name="pv_iv_uncertainty_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


st.info(
    """
    **Notes:**
    * Rectangular inputs -> $a/\\sqrt{3}$; Normal inputs are $\\sigma$.
    * Temperature effects use $u(y)^2 = (\\DeltaT)^2*u(TC)^2 + (TC)^2*u(\\DeltaT)^2$.
    * Temperature deviations converted to relative % via $(\\text{dev}/25)*100$.
    * Pmpp uncertainty assumes independence between Impp and Vmpp: $\\sqrt{u_{\\text{rel}}(I)^2 + u_{\\text{rel}}(V)^2}$.
    * Geometry/orientation items are modeled as % contributors; detailed optical model can refine them.
    * Lead resistance not auto-converted to % in this version.
    """
)
