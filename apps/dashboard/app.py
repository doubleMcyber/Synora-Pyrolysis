from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from synora.calibration.surrogate_fit import (
    DEFAULT_PARAMS_PATH,
    calibrate_and_store,
    calibrated_predict,
    load_pfr_dataset,
)
from synora.economics.lcoh import EconInputs
from synora.generative.design_space import ReactorDesign
from synora.generative.objectives import evaluate_design_surrogate
from synora.generative.optimizer import evaluations_to_frame, propose_designs
from synora.twin.simulator import run_simulation

st.set_page_config(page_title="Synora Twin Console", layout="wide")

st.markdown(
    """
<style>
.stApp {
    background-color: #0b1220;
    color: #e5edf7;
    font-family: "JetBrains Mono", "SFMono-Regular", monospace;
}
[data-testid="stSidebar"] {
    background-color: #0e1627;
    border-right: 1px solid #1f2a3f;
}
[data-testid="stMetric"] {
    background-color: #111b2f;
    border: 1px solid #2a3b58;
    border-radius: 8px;
    padding: 10px 12px;
}
h1, h2, h3 {
    color: #dce7ff;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOT_TEMPLATE = "plotly_dark"
ACCENT_BLUE = "#4c8dff"
ACCENT_GREEN = "#33d17a"
ACCENT_ORANGE = "#f6a04d"
ACCENT_PURPLE = "#a371f7"


@st.cache_data(show_spinner=False)
def _load_latest_physics_dataset() -> pd.DataFrame | None:
    try:
        return load_pfr_dataset()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def _ensure_calibrated_params() -> str | None:
    try:
        calibrate_and_store(verbose=False)
        return str(DEFAULT_PARAMS_PATH)
    except FileNotFoundError:
        return None


def _predict_frame(
    temperatures_c: np.ndarray, residence_time_s: float, params_path: str | None
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for temp_c in temperatures_c:
        pred = calibrated_predict(float(temp_c), residence_time_s, params_path=params_path)
        rows.append(
            {
                "temperature_c": float(temp_c),
                "methane_conversion": pred["methane_conversion"],
                "h2_yield_mol_per_mol_ch4": pred["h2_yield_mol_per_mol_ch4"],
                "carbon_formation_index": pred["carbon_formation_index"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _profit_envelope(
    *,
    intervals_hr: tuple[int, ...],
    hours: int,
    methane_kg_per_hr: float,
    temp_c: float,
    residence_time_s: float,
    econ_inputs: EconInputs,
    params_path: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for interval in intervals_hr:
        simulation = run_simulation(
            hours=hours,
            methane_kg_per_hr=methane_kg_per_hr,
            temp=temp_c,
            residence_time_s=residence_time_s,
            econ_inputs=econ_inputs,
            maintenance_interval_hr=interval,
            surrogate_params_path=params_path,
        )
        rows.append(
            {
                "maintenance_interval_hr": float(interval),
                "cum_profit_usd": float(simulation["cum_profit_usd"].iloc[-1]),
                "mean_profit_per_hr": float(simulation["profit_per_hr"].mean()),
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Twin Controls")
    hours = st.slider("Simulation horizon (hours)", min_value=12, max_value=336, value=72, step=12)
    methane_kg_per_hr = st.slider(
        "Methane feed (kg/hr)",
        min_value=20,
        max_value=300,
        value=120,
        step=5,
    )
    temp_c = st.slider(
        "Reactor temperature (degC)", min_value=850, max_value=1100, value=980, step=10
    )
    residence_time_s = st.slider(
        "Residence time (s)", min_value=0.1, max_value=5.0, value=1.5, step=0.1
    )
    maintenance_interval_hr = st.slider(
        "Maintenance interval (hr)", min_value=12, max_value=240, value=72, step=12
    )

    st.header("Economics")
    hydrogen_price = st.number_input("Hydrogen price (USD/kg)", min_value=0.0, value=4.7, step=0.1)
    carbon_price = st.number_input("Carbon price (USD/kg)", min_value=0.0, value=0.22, step=0.01)
    methane_price = st.number_input("Methane price (USD/kg)", min_value=0.0, value=0.48, step=0.01)
    variable_opex = st.number_input("Variable opex (USD/hr)", min_value=0.0, value=18.0, step=1.0)
    fixed_opex = st.number_input("Fixed opex (USD/hr)", min_value=0.0, value=10.0, step=1.0)

econ_inputs = EconInputs(
    methane_price_usd_per_kg=float(methane_price),
    hydrogen_price_usd_per_kg=float(hydrogen_price),
    carbon_price_usd_per_kg=float(carbon_price),
    variable_opex_usd_per_hr=float(variable_opex),
    fixed_opex_usd_per_hr=float(fixed_opex),
)

physics_df = _load_latest_physics_dataset()
params_path = _ensure_calibrated_params()

results = run_simulation(
    hours=hours,
    methane_kg_per_hr=float(methane_kg_per_hr),
    temp=float(temp_c),
    residence_time_s=float(residence_time_s),
    econ_inputs=econ_inputs,
    maintenance_interval_hr=int(maintenance_interval_hr),
    surrogate_params_path=params_path,
)
latest = results.iloc[-1]

st.title("Synora Methane Pyrolysis Twin Console")
st.caption("Phase B/C: Physics dataset + fouling-aware health + calibrated surrogate twin")

if physics_df is None:
    st.warning(
        "No physics parquet found in data/processed/physics_runs. "
        "Run scripts/physics/generate_pfr_dataset.py to enable physics overlays."
    )
elif params_path is None:
    st.warning("Physics data found, but surrogate params were not generated.")

tab_overview, tab_sensitivity, tab_envelope, tab_design = st.tabs(
    ["Reactor Overview", "Sensitivity", "Economic Envelope", "Design Explorer"]
)

with tab_overview:
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("LCOH (USD/kg)", f"{latest['lcoh_usd_per_kg']:.2f}")
    kpi_cols[1].metric("H2 Output (kg/hr)", f"{latest['h2_kg_per_hr']:.2f}")
    kpi_cols[2].metric("Carbon Rate (kg/hr)", f"{latest['carbon_rate_kg_per_hr']:.2f}")
    kpi_cols[3].metric("Reactor Health", f"{latest['health']:.3f}")

    if physics_df is not None:
        filtered = physics_df.copy()
        filtered["tau_distance"] = (filtered["residence_time_s"] - float(residence_time_s)).abs()
        nearest_tau = float(filtered.sort_values("tau_distance").iloc[0]["residence_time_s"])
        physics_slice = (
            filtered[filtered["residence_time_s"] == nearest_tau]
            .sort_values("temperature_c")
            .reset_index(drop=True)
        )
        surrogate_slice = _predict_frame(
            physics_slice["temperature_c"].to_numpy(dtype=float),
            residence_time_s=float(residence_time_s),
            params_path=params_path,
        )

        overview_cols = st.columns(2)

        conv_fig = go.Figure()
        conv_fig.add_trace(
            go.Scatter(
                x=physics_slice["temperature_c"],
                y=physics_slice["methane_conversion"],
                mode="markers",
                name=f"Physics (tau={nearest_tau:.2f}s)",
                marker=dict(color=ACCENT_ORANGE, size=8),
            )
        )
        conv_fig.add_trace(
            go.Scatter(
                x=surrogate_slice["temperature_c"],
                y=surrogate_slice["methane_conversion"],
                mode="lines",
                name="Surrogate",
                line=dict(color=ACCENT_BLUE, width=2.5),
            )
        )
        conv_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="Methane Conversion: Physics vs Surrogate",
            xaxis_title="Temperature (degC)",
            yaxis_title="Conversion",
            legend_title="Model",
        )

        h2_fig = go.Figure()
        h2_fig.add_trace(
            go.Scatter(
                x=physics_slice["temperature_c"],
                y=physics_slice["h2_yield_mol_per_mol_ch4"],
                mode="markers",
                name=f"Physics (tau={nearest_tau:.2f}s)",
                marker=dict(color=ACCENT_ORANGE, size=8),
            )
        )
        h2_fig.add_trace(
            go.Scatter(
                x=surrogate_slice["temperature_c"],
                y=surrogate_slice["h2_yield_mol_per_mol_ch4"],
                mode="lines",
                name="Surrogate",
                line=dict(color=ACCENT_GREEN, width=2.5),
            )
        )
        h2_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="H2 Yield: Physics vs Surrogate",
            xaxis_title="Temperature (degC)",
            yaxis_title="H2 Yield (mol/mol CH4)",
            legend_title="Model",
        )

        overview_cols[0].plotly_chart(conv_fig, width="stretch")
        overview_cols[1].plotly_chart(h2_fig, width="stretch")

    runtime_cols = st.columns(2)
    production_fig = go.Figure()
    production_fig.add_trace(
        go.Scatter(
            x=results["time_hr"],
            y=results["h2_kg_per_hr"],
            mode="lines",
            name="H2 rate",
            line=dict(color=ACCENT_GREEN, width=2.2),
        )
    )
    production_fig.add_trace(
        go.Scatter(
            x=results["time_hr"],
            y=results["carbon_rate_kg_per_hr"],
            mode="lines",
            name="Carbon generation rate",
            line=dict(color=ACCENT_ORANGE, width=2.2),
        )
    )
    production_fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Twin Runtime: Product and Carbon Rates",
        xaxis_title="Time (hr)",
        yaxis_title="kg/hr",
    )
    runtime_cols[0].plotly_chart(production_fig, width="stretch")

    health_fig = go.Figure()
    health_fig.add_trace(
        go.Scatter(
            x=results["time_hr"],
            y=results["health"],
            mode="lines",
            name="Health",
            line=dict(color=ACCENT_PURPLE, width=2.2),
        )
    )
    health_fig.add_trace(
        go.Scatter(
            x=results["time_hr"],
            y=results["fouling_rate_per_hr"],
            mode="lines",
            name="Fouling rate",
            line=dict(color=ACCENT_BLUE, width=2.0, dash="dot"),
            yaxis="y2",
        )
    )
    health_fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Health and Fouling Dynamics",
        xaxis_title="Time (hr)",
        yaxis=dict(title="Health"),
        yaxis2=dict(title="Fouling rate (1/hr)", overlaying="y", side="right"),
    )
    runtime_cols[1].plotly_chart(health_fig, width="stretch")

with tab_sensitivity:
    temperatures_c = np.linspace(850.0, 1100.0, 35)
    taus_s = np.linspace(0.1, 5.0, 35)
    z_matrix = np.zeros((len(taus_s), len(temperatures_c)))
    for tau_idx, tau in enumerate(taus_s):
        for temp_idx, temp in enumerate(temperatures_c):
            pred = calibrated_predict(float(temp), float(tau), params_path=params_path)
            z_matrix[tau_idx, temp_idx] = pred["methane_conversion"]

    heatmap_fig = go.Figure(
        data=go.Heatmap(
            x=temperatures_c,
            y=taus_s,
            z=z_matrix,
            colorscale="Turbo",
            colorbar=dict(title="Conversion"),
        )
    )
    heatmap_fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Temperature vs Conversion Sensitivity",
        xaxis_title="Temperature (degC)",
        yaxis_title="Residence time (s)",
    )
    st.plotly_chart(heatmap_fig, width="stretch")

with tab_envelope:
    envelope_intervals = tuple(range(12, 241, 12))
    envelope_df = _profit_envelope(
        intervals_hr=envelope_intervals,
        hours=int(hours),
        methane_kg_per_hr=float(methane_kg_per_hr),
        temp_c=float(temp_c),
        residence_time_s=float(residence_time_s),
        econ_inputs=econ_inputs,
        params_path=params_path,
    )

    envelope_fig = go.Figure()
    envelope_fig.add_trace(
        go.Scatter(
            x=envelope_df["maintenance_interval_hr"],
            y=envelope_df["cum_profit_usd"],
            mode="lines+markers",
            name="Cumulative profit",
            line=dict(color=ACCENT_GREEN, width=2.5),
            marker=dict(size=8),
        )
    )
    envelope_fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Profit vs Maintenance Interval",
        xaxis_title="Maintenance interval (hr)",
        yaxis_title="Cumulative profit (USD)",
    )
    st.plotly_chart(envelope_fig, width="stretch")
    st.dataframe(envelope_df, width="stretch")

with tab_design:
    st.subheader("Generative Design Explorer")
    st.caption(
        "Evolutionary search on surrogate objectives. Physics labeling remains offline via active learning."
    )

    control_cols = st.columns(4)
    top_k = control_cols[0].slider("Top K designs", min_value=5, max_value=40, value=15, step=1)
    generations = control_cols[1].slider("Generations", min_value=4, max_value=30, value=10, step=1)
    population = control_cols[2].slider(
        "Population size", min_value=40, max_value=300, value=120, step=10
    )
    optimizer_seed = control_cols[3].number_input(
        "Optimizer seed", min_value=1, max_value=100000, value=42, step=1
    )

    constraint_cols = st.columns(4)
    max_fouling = constraint_cols[0].slider(
        "Max Fouling Risk Index", min_value=0.05, max_value=1.0, value=0.40, step=0.01
    )
    max_dp = constraint_cols[1].slider(
        "Max pressure-drop proxy", min_value=0.05, max_value=1.0, value=0.35, step=0.01
    )
    max_heat_loss = constraint_cols[2].slider(
        "Max heat-loss proxy", min_value=0.05, max_value=1.0, value=0.40, step=0.01
    )
    min_conversion = constraint_cols[3].slider(
        "Min conversion", min_value=0.0, max_value=0.3, value=0.01, step=0.005
    )

    if st.button("Run Design Optimization", type="primary"):
        constraints = {
            "min_conversion": float(min_conversion),
            "max_fouling_risk_index": float(max_fouling),
            "max_pressure_drop_proxy": float(max_dp),
            "max_heat_loss_proxy": float(max_heat_loss),
        }
        evaluations = propose_designs(
            top_k=int(top_k),
            generations=int(generations),
            population_size=int(population),
            seed=int(optimizer_seed),
            surrogate_params_path=params_path,
            econ_inputs=econ_inputs,
            constraints=constraints,
        )
        st.session_state["design_leaderboard"] = evaluations_to_frame(evaluations)

    leaderboard = st.session_state.get("design_leaderboard", pd.DataFrame())
    if leaderboard.empty:
        st.info("Run optimization to generate candidate reactor designs.")
    else:
        display_columns = [
            "score",
            "profit_per_hr",
            "conversion",
            "h2_rate",
            "fouling_risk_index",
            "pressure_drop_proxy",
            "heat_loss_proxy",
            "constraint_violation_count",
            "temp_c",
            "residence_time_s",
            "length_m",
            "diameter_m",
            "pressure_atm",
            "methane_kg_per_hr",
            "dilution_frac",
            "carbon_removal_eff",
        ]
        available_columns = [column for column in display_columns if column in leaderboard.columns]
        ranked = leaderboard.sort_values(
            ["constraint_violation_count", "score"], ascending=[True, False]
        )

        st.markdown("**Leaderboard**")
        st.dataframe(ranked[available_columns], width="stretch")

        pareto_cols = st.columns(2)
        pareto_fig = go.Figure()
        pareto_fig.add_trace(
            go.Scatter(
                x=ranked["fouling_risk_index"],
                y=ranked["profit_per_hr"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=ranked["conversion"],
                    colorscale="Turbo",
                    colorbar=dict(title="Conversion"),
                ),
                text=[f"Design {idx}" for idx in ranked.index],
                hovertemplate=(
                    "Fouling risk: %{x:.3f}<br>"
                    "Profit/hr: %{y:.2f}<br>"
                    "Conversion: %{marker.color:.3f}<extra></extra>"
                ),
                name="Designs",
            )
        )
        pareto_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="Pareto: Profit vs Fouling Risk",
            xaxis_title="Fouling Risk Index",
            yaxis_title="Profit per hour (USD)",
        )
        pareto_cols[0].plotly_chart(pareto_fig, width="stretch")

        pareto_fig_2 = go.Figure()
        pareto_fig_2.add_trace(
            go.Scatter(
                x=ranked["conversion"],
                y=ranked["profit_per_hr"],
                mode="markers",
                marker=dict(size=9, color=ACCENT_BLUE),
                hovertemplate="Conversion: %{x:.3f}<br>Profit/hr: %{y:.2f}<extra></extra>",
                name="Designs",
            )
        )
        pareto_fig_2.update_layout(
            template=PLOT_TEMPLATE,
            title="Pareto: Profit vs Conversion",
            xaxis_title="Conversion",
            yaxis_title="Profit per hour (USD)",
        )
        pareto_cols[1].plotly_chart(pareto_fig_2, width="stretch")

        ranked = ranked.reset_index(drop=True)
        selected_idx = st.selectbox(
            "Inspect design candidate",
            options=list(range(len(ranked))),
            format_func=lambda idx: (
                f"#{idx + 1} score={ranked.loc[idx, 'score']:.2f}, "
                f"profit={ranked.loc[idx, 'profit_per_hr']:.2f}, "
                f"fouling={ranked.loc[idx, 'fouling_risk_index']:.3f}"
            ),
        )
        selected = ranked.loc[selected_idx]
        selected_design = ReactorDesign(
            length_m=float(selected["length_m"]),
            diameter_m=float(selected["diameter_m"]),
            pressure_atm=float(selected["pressure_atm"]),
            temp_c=float(selected["temp_c"]),
            methane_kg_per_hr=float(selected["methane_kg_per_hr"]),
            dilution_frac=float(selected["dilution_frac"]),
            carbon_removal_eff=float(selected["carbon_removal_eff"]),
            wall_thickness_m=float(selected["wall_thickness_m"]),
            emissivity=float(selected["emissivity"]),
            roughness_mm=float(selected["roughness_mm"]),
        )
        selected_metrics = evaluate_design_surrogate(
            selected_design,
            surrogate_params_path=params_path,
            econ_inputs=econ_inputs,
        )
        detail_cols = st.columns(4)
        detail_cols[0].metric(
            "Selected Profit/hr (USD)", f"{float(selected_metrics['profit_per_hr']):.2f}"
        )
        detail_cols[1].metric("Selected Conversion", f"{float(selected_metrics['conversion']):.3f}")
        detail_cols[2].metric(
            "Fouling Risk Index", f"{float(selected_metrics['fouling_risk_index']):.3f}"
        )
        detail_cols[3].metric("Residence Time (s)", f"{selected_design.residence_time_s:.2f}")

        candidate_sim = run_simulation(
            hours=hours,
            methane_kg_per_hr=selected_design.methane_kg_per_hr,
            temp=selected_design.temp_c,
            residence_time_s=selected_design.residence_time_s,
            econ_inputs=econ_inputs,
            maintenance_interval_hr=int(maintenance_interval_hr),
            surrogate_params_path=params_path,
        )
        sim_fig = go.Figure()
        sim_fig.add_trace(
            go.Scatter(
                x=candidate_sim["time_hr"],
                y=candidate_sim["profit_per_hr"],
                mode="lines",
                name="Profit/hr",
                line=dict(color=ACCENT_GREEN, width=2.2),
            )
        )
        sim_fig.add_trace(
            go.Scatter(
                x=candidate_sim["time_hr"],
                y=candidate_sim["health"],
                mode="lines",
                name="Health",
                line=dict(color=ACCENT_PURPLE, width=2.2, dash="dot"),
                yaxis="y2",
            )
        )
        sim_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="Selected Candidate: Profit and Health Over Time",
            xaxis_title="Time (hr)",
            yaxis=dict(title="Profit/hr (USD)"),
            yaxis2=dict(title="Health", overlaying="y", side="right"),
        )
        st.plotly_chart(sim_fig, width="stretch")
