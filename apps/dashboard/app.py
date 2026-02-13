import plotly.express as px
import streamlit as st

from synora.economics.lcoh import EconInputs
from synora.twin.simulator import run_simulation

st.set_page_config(page_title="Synora", layout="wide")
st.title("Synora Layer 1 - Reactor/Economics Twin")
st.caption("Deterministic methane pyrolysis reactor simulation with hourly economics.")

with st.sidebar:
    st.header("Reactor")
    hours = st.slider("Simulation horizon (hours)", min_value=6, max_value=168, value=24, step=6)
    methane_kg_per_hr = st.slider(
        "Methane feed (kg/hr)",
        min_value=10,
        max_value=250,
        value=100,
        step=5,
    )
    temp_c = st.slider(
        "Reactor temperature (degC)", min_value=650, max_value=1000, value=800, step=10
    )

    st.header("Economics")
    hydrogen_price = st.number_input("Hydrogen price (USD/kg)", min_value=0.0, value=4.5, step=0.1)
    carbon_price = st.number_input("Carbon price (USD/kg)", min_value=0.0, value=0.18, step=0.01)
    methane_price = st.number_input("Methane price (USD/kg)", min_value=0.0, value=0.45, step=0.01)
    variable_opex = st.number_input("Variable opex (USD/hr)", min_value=0.0, value=15.0, step=1.0)
    fixed_opex = st.number_input("Fixed opex (USD/hr)", min_value=0.0, value=8.0, step=1.0)

econ_inputs = EconInputs(
    methane_price_usd_per_kg=float(methane_price),
    hydrogen_price_usd_per_kg=float(hydrogen_price),
    carbon_price_usd_per_kg=float(carbon_price),
    variable_opex_usd_per_hr=float(variable_opex),
    fixed_opex_usd_per_hr=float(fixed_opex),
)
results = run_simulation(
    hours=hours,
    methane_kg_per_hr=float(methane_kg_per_hr),
    temp=float(temp_c),
    econ_inputs=econ_inputs,
)
latest = results.iloc[-1]

kpi_cols = st.columns(4)
kpi_cols[0].metric("H2 output (kg/hr)", f"{latest['h2_kg_per_hr']:.2f}")
kpi_cols[1].metric("Carbon output (kg/hr)", f"{latest['carbon_kg_per_hr']:.2f}")
kpi_cols[2].metric("Profit (USD/hr)", f"{latest['profit_per_hr']:.2f}")
kpi_cols[3].metric("LCOH (USD/kg)", f"{latest['lcoh_usd_per_kg']:.2f}")

chart_cols = st.columns(2)
production_fig = px.line(
    results,
    x="time_hr",
    y=["h2_kg_per_hr", "carbon_kg_per_hr"],
    title="Hourly Production",
)
economics_fig = px.line(
    results,
    x="time_hr",
    y=["profit_per_hr", "lcoh_usd_per_kg"],
    title="Hourly Economics",
)
chart_cols[0].plotly_chart(production_fig, width="stretch")
chart_cols[1].plotly_chart(economics_fig, width="stretch")

st.subheader("Simulation Results")
st.dataframe(results, width="stretch")
