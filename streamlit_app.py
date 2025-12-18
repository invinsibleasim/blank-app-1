
# -*- coding: utf-8 -*-
# Streamlit app: PV Module IV Measurement Uncertainty Calculator
# File: pv_iv_uncertainty_app.py

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

DIST_RECTANGULAR = "Rectangular"
DIST_NORMAL = "Normal"

def standard_uncertainty(value: float, distribution: str) -> float:
    # Rectangular: standard uncertainty is a/sqrt(3); Normal: value is already sigma
    if distribution == DIST_RECTANGULAR:
        return float(value) / math.sqrt(3)
    return float(value)

def combine_rss(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    return float(np.sqrt(np.sum(arr ** 2)))

def u_temp_effect(tc_percent_per_k: float, u_tc_percent_per_k: float, dT_k: float, u_dT_k: float) -> float:
    # u(y)^2 = (dT)^2 * u(TC)^2 + (TC)^2 * u(dT)^2
    return float(np.sqrt((dT_k ** 2) * (u_tc_percent_per_k ** 2) + (tc_percent_per_k ** 2) * (u_dT_k ** 2)))

def percent_from_temp_dev_c(dev_c: float) -> float:
    # (dev / 25 degC) * 100 converts to relative percent
    return float(dev_c) / 25.0 * 100.0

st.set_page_config(page_title="PV IV Measurement Uncertainty", layout="wide")
st.title("PV Module IV Measurement Uncertainty Calculator")
st.caption("Compute combined standard uncertainty and expanded uncertainty for IV parameters (Isc, Impp, Voc, Vmpp, Pmpp) using a configurable GUM-based budget.")

with st.sidebar:
    st.header("Global settings")
    k = st.number_input("Coverage factor k (approx 2 for ~95%)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    st.divider()

    st.subheader("Environmental conditions")
    t_module = st.number_input("Measured module temperature [degC]", value=25.0, step=0.1)
    t_ref = st.number_input("Reference temperature [degC] (STC=25)", value=25.0, step=0.1)
    dT = t_module - t_ref
    st.write(f"Delta T = {dT:.2f} degC")

    max_temp_dev = st.number_input("Max temperature deviation [degC] (half-width a)", value=1.0, step=0.1)
    u_dT = standard_uncertainty(max_temp_dev, DIST_RECTANGULAR)
    st.write(f"Standard uncertainty u(Delta T) = {u_dT:.3f} degC")

    irr_nonuni = st.number_input("Max irradiance non-uniformity [%] (half-width a)", value=1.6, step=0.1)
    irr_dev_ref = st.number_input("Irradiance deviation module->reference cell [%] (half-width a)", value=0.50, step=0.01)

    st.subheader("Spectral mismatch")
    mm_sigma = st.number_input("Spectral MM uncertainty [%] (sigma)", value=1.00, step=0.01)

    st.subheader("Temperature coefficients")
    tc_i = st.number_input("TC_I [%/K]", value=0.04, step=0.01)
    tc_i_dev_max = st.number_input("Max. TC_I deviation [%/K] (half-width a)", value=0.02, step=0.01)
    u_tc_i = standard_uncertainty(tc_i_dev_max, DIST_RECTANGULAR)

    tc_u = st.number_input("TC_U [%/K]", value=0.40, step=0.01)
    tc_u_dev_max = st.number_input("Max. TC_U deviation [%/K] (half-width a)", value=0.10, step=0.01)
    u_tc_u = standard_uncertainty(tc_u_dev_max, DIST_RECTANGULAR)

    st.subheader("Temperature measurement chain")
    temp_dev_backsheet = st.number_input("Temp deviation backsheet<->pn-junction [degC] (half-width a)", value=0.40, step=0.05)
    temp_sensors_mean_dev = st.number_input("Mean of sensors <-> true mean [degC] (half-width a)", value=0.40, step=0.05)
    include_temp_chain = st.checkbox("Include temp measurement-chain contributions in current-related parameters", value=True)

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

# Derived uncertainties
u_mm = standard_uncertainty(mm_sigma, DIST_NORMAL)
u_irr_nonuni = standard_uncertainty(irr_nonuni, DIST_RECTANGULAR)
u_irr_dev_ref = standard_uncertainty(irr_dev_ref, DIST_RECTANGULAR)

# Temperature chain to %
temp_chain_percent_1 = percent_from_temp_dev_c(temp_dev_backsheet)
temp_chain_percent_2 = percent_from_temp_dev_c(temp_sensors_mean_dev)
u_temp_chain_1 = standard_uncertainty(temp_chain_percent_1, DIST_RECTANGULAR)
u_temp_chain_2 = standard_uncertainty(temp_chain_percent_2, DIST_RECTANGULAR)

# Temperature-induced relative uncertainty for current and voltage
nu_temp_I = u_temp_effect(tc_i, u_tc_i, dT, u_dT)
nu_temp_U = u_temp_effect(tc_u, u_tc_u, dT, u_dT)

# Hysteresis (rectangular)
nu_hyst_isc = standard_uncertainty(hyst_isc, DIST_RECTANGULAR)
nu_hyst_impp = standard_uncertainty(hyst_impp, DIST_RECTANGULAR)
nu_hyst_voc = standard_uncertainty(hyst_voc, DIST_RECTANGULAR)
nu_hyst_vmpp = standard_uncertainty(hyst_vmpp, DIST_RECTANGULAR)
nu_hyst_pmpp = standard_uncertainty(hyst_pmpp, DIST_RECTANGULAR)

# Reproducibility (normal)
nu_repro_isc = standard_uncertainty(repro_isc, DIST_NORMAL)
nu_repro_impp = standard_uncertainty(repro_impp, DIST_NORMAL)
nu_repro_voc = standard_uncertainty(repro_voc, DIST_NORMAL)
nu_repro_vmpp = standard_uncertainty(repro_vmpp, DIST_NORMAL)
nu_repro_pmpp = standard_uncertainty(repro_pmpp, DIST_NORMAL)

# Build contribution dicts
contrib_I_common = {
    "Spectral mismatch (MM)": u_mm,
    "Irradiance non-uniformity": u_irr_nonuni,
    "Irradiance dev. module->ref cell": u_irr_dev_ref,
}
if include_temp_chain:
    contrib_I_common.update({
        "Temp chain: backsheet<->pn junction": u_temp_chain_1,
        "Temp chain: sensors mean<->true mean": u_temp_chain_2,
    })
contrib_I_current = {
    "Temperature effect via TC_I": u_temp_I,
    "Hysteresis": u_hyst_isc,
    "Reproducibility": u_repro_isc,
}
contrib_Impp_current = {
    "Temperature effect via TC_I": u_temp_I,
    "Hysteresis": u_hyst_impp,
    "Reproducibility": u_repro_impp,
}
contrib_V_common = {"Temperature effect via TC_U": u_temp_U}
if include_irr_in_voltage:
    contrib_V_common.update({
        "Irradiance non-uniformity": u_irr_nonuni,
        "Irradiance dev. module->ref cell": u_irr_dev_ref,
        "Spectral mismatch (MM)": u_mm,
    })
contrib_V_voc = {"Hysteresis": u_hyst_voc, "Reproducibility": u_repro_voc}
contrib_V_vmpp = {"Hysteresis": u_hyst_vmpp, "Reproducibility": u_repro_vmpp}
contrib_P_pmpp_specific = {"Hysteresis": u_hyst_pmpp, "Reproducibility": u_repro_pmpp}

# Compute budgets

def make_budget_df(contribs: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame({"Contribution": list(contribs.keys()), "Standard uncertainty u_j [%]": list(contribs.values())})
    df["u_j^2"] = df["Standard uncertainty u_j [%]"] ** 2
    return df

contribs_isc = {**contrib_I_common, **contrib_I_current}
isc_df = make_budget_df(contribs_isc)
uc_isc = combine_rss(list(contribs_isc.values()))
U_isc = k * uc_isc

contribs_impp = {**contrib_I_common, **contrib_Impp_current}
impp_df = make_budget_df(contribs_impp)
uc_impp = combine_rss(list(contribs_impp.values()))
U_impp = k * uc_impp

contribs_voc = {**contrib_V_common, **contrib_V_voc}
voc_df = make_budget_df(contribs_voc)
uc_voc = combine_rss(list(contribs_voc.values()))
U_voc = k * uc_voc

contribs_vmpp = {**contrib_V_common, **contrib_V_vmpp}
vmpp_df = make_budget_df(contribs_vmpp)
uc_vmpp = combine_rss(list(contribs_vmpp.values()))
U_vmpp = k * uc_vmpp

uc_pmpp_from_iv = np.sqrt(uc_impp ** 2 + uc_vmpp ** 2)
contribs_pmpp = {"From Impp & Vmpp": uc_pmpp_from_iv, **contrib_P_pmpp_specific}
pmpp_df = make_budget_df(contribs_pmpp)
uc_pmpp = combine_rss(list(contribs_pmpp.values()))
U_pmpp = k * uc_pmpp

st.subheader("Uncertainty budgets (standard uncertainties in %)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Isc")
    st.dataframe(isc_df, use_container_width=True)
    st.write(f"Combined standard uncertainty uc(Isc) = **{uc_isc:.3f}%**; Expanded U (k={k}) = **{U_isc:.3f}%**")

    st.markdown("### Voc")
    st.dataframe(voc_df, use_container_width=True)
    st.write(f"Combined standard uncertainty uc(Voc) = **{uc_voc:.3f}%**; Expanded U (k={k}) = **{U_voc:.3f}%**")

with col2:
    st.markdown("### Impp")
    st.dataframe(impp_df, use_container_width=True)
    st.write(f"Combined standard uncertainty uc(Impp) = **{uc_impp:.3f}%**; Expanded U (k={k}) = **{U_impp:.3f}%**")

    st.markdown("### Vmpp")
    st.dataframe(vmpp_df, use_container_width=True)
    st.write(f"Combined standard uncertainty uc(Vmpp) = **{uc_vmpp:.3f}%**; Expanded U (k={k}) = **{U_vmpp:.3f}%**")

st.markdown("### Pmpp")
st.dataframe(pmpp_df, use_container_width=True)
st.write(f"Combined standard uncertainty uc(Pmpp) = **{uc_pmpp:.3f}%**; Expanded U (k={k}) = **{U_pmpp:.3f}%**")

st.subheader("Contribution breakdown")
tabs = st.tabs(["Isc", "Impp", "Voc", "Vmpp", "Pmpp"])
for tab, df in zip(tabs, [isc_df, impp_df, voc_df, vmpp_df, pmpp_df]):
    with tab:
        plot_df = df.sort_values("Standard uncertainty u_j [%]", ascending=True)
        st.bar_chart(plot_df.set_index("Contribution")[["Standard uncertainty u_j [%]"]])

st.subheader("Export uncertainty report")
all_tables = {"Isc": isc_df, "Impp": impp_df, "Voc": voc_df, "Vmpp": vmpp_df, "Pmpp": pmpp_df}
summary = pd.DataFrame({
    "Parameter": ["Isc", "Impp", "Voc", "Vmpp", "Pmpp"],
    "uc [%]": [uc_isc, uc_impp, uc_voc, uc_vmpp, uc_pmpp],
    "k": [k] * 5,
    "U = k·uc [%]": [U_isc, U_impp, U_voc, U_vmpp, U_pmpp],
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

st.info('''
Notes
- Rectangular distributions -> standard uncertainties via a/sqrt(3).
- Temperature effects: u(y)^2 = (DeltaT)^2 * u(TC)^2 + (TC)^2 * u(DeltaT)^2.
- Temperature measurement-chain deviations -> relative % using (dev / 25 degC) * 100.
- Pmpp uncertainty assumes independence between Impp and Vmpp contributions (u_rel(P) approx sqrt(u_rel(I)^2 + u_rel(V)^2)).
- Adjust inputs in the sidebar to match your measurement setup.
''')
