from __future__ import annotations

import importlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reactor_3d_component import render_reactor_3d

from synora import __version__ as SYNORA_VERSION
from synora.calibration.surrogate_fit import (
    DEFAULT_PARAMS_PATH,
    DEFAULT_PHYSICS_DIR,
    calibrate_and_store,
    calibrated_predict,
    latest_physics_dataset,
    load_pfr_dataset,
)
from synora.economics.lcoh import EconInputs
from synora.generative.design_space import ReactorDesign
from synora.generative.multizone import MultiZoneBounds, MultiZoneDesign, ZoneDesign
from synora.generative.objectives import evaluate_design_surrogate, evaluate_multizone_surrogate
from synora.generative.optimizer import (
    evaluations_to_frame,
    multizone_evaluations_to_frame,
    propose_designs,
    propose_multizone_designs,
)
from synora.physics.label_pfr import label_pfr_case
from synora.twin.simulator import build_simulation_context, build_visual_frame, run_simulation
from synora.validation.experimental import (
    DEFAULT_EXPERIMENTAL_DATASET_PATH,
    VALIDATION_AXIS_NOTE,
    load_cv_reactor_experiment,
)
from synora.validation.metrics import compare_experiment_to_surrogate

st.set_page_config(
    page_title="Synora Industrial Twin",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp {
    background:
        linear-gradient(rgba(18, 27, 42, 0.92), rgba(9, 14, 23, 0.95)),
        repeating-linear-gradient(
            to right,
            rgba(160, 180, 210, 0.04) 0,
            rgba(160, 180, 210, 0.04) 1px,
            transparent 1px,
            transparent 32px
        ),
        repeating-linear-gradient(
            to bottom,
            rgba(160, 180, 210, 0.04) 0,
            rgba(160, 180, 210, 0.04) 1px,
            transparent 1px,
            transparent 32px
        );
    color: #d9e6f8;
    font-family: "Inter", "IBM Plex Sans", "Roboto", sans-serif;
}
[data-testid="stAppViewContainer"] * {
    font-family: "Inter", "IBM Plex Sans", "Roboto", sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0c131f;
    border-right: 1px solid #1f2a3f;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    font-family: "Roboto Mono", "SFMono-Regular", "Consolas", monospace;
}
[data-testid="stNumberInput"] input {
    font-family: "Roboto Mono", "SFMono-Regular", "Consolas", monospace !important;
}
[data-testid="stExpander"], [data-testid="stNumberInput"], [data-testid="stSlider"] {
    border-color: rgba(92, 116, 150, 0.45) !important;
}
[data-testid="stExpander"] summary {
    min-height: 2.05rem !important;
    align-items: center !important;
}
[data-testid="stExpander"] summary p {
    margin: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    letter-spacing: 0.05em;
}
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(18, 29, 47, 0.95), rgba(12, 20, 34, 0.95));
    border: 1px solid rgba(92, 116, 150, 0.55);
    border-radius: 2px;
    padding: 8px 10px;
    box-shadow: 0 0 0 rgba(76,141,255,0.0);
    animation: metricPulse 0.9s ease-out;
}
h1, h2, h3 {
    color: #dce7ff;
    font-family: "IBM Plex Sans", "Inter", sans-serif;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 500;
}
.stTabs [data-baseweb="tab-list"] { gap: 0.35rem; }
.stTabs [data-baseweb="tab"] {
    background: rgba(10, 18, 30, 0.7);
    border: 1px solid rgba(87, 107, 139, 0.45);
    border-radius: 2px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.75rem;
}
.stTabs [aria-selected="true"] {
    border-color: rgba(88, 161, 255, 0.8) !important;
    box-shadow: 0 0 12px rgba(50, 140, 255, 0.28);
}
.kpi-card-label {
    color: #b6c9e2;
    font-size: 0.74rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.kpi-card-value {
    color: #e3eeff;
    font-family: "Roboto Mono", "SFMono-Regular", "Consolas", monospace;
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 0.08rem;
}
.kpi-card-stack {
    width: 100%;
    text-align: right;
}
.kpi-card-trend {
    margin-top: 0.08rem;
    font-size: 0.84rem;
    letter-spacing: 0.06em;
    font-weight: 600;
}
.design-hero-header {
    color: #9fc3ea;
    font-size: 0.80rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.design-hero-subheader {
    color: #dce9ff;
    font-size: 0.94rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.design-hero-kpi-title {
    color: #9fb8d4;
    font-size: 0.68rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}
.design-hero-kpi-value {
    color: #eef5ff;
    font-family: "Roboto Mono", "SFMono-Regular", "Consolas", monospace;
    font-size: 1.60rem;
    font-weight: 600;
}
.st-emotion-cache-1f5xw1n, .st-emotion-cache-13k62yr {
    border-radius: 2px !important;
}
@keyframes metricPulse {
  0% { box-shadow: 0 0 0 rgba(76,141,255,0.0); }
  30% { box-shadow: 0 0 14px rgba(76,141,255,0.35); }
  100% { box-shadow: 0 0 0 rgba(76,141,255,0.0); }
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
ACCENT_RED = "#ff4d5f"
GRID_COLOR = "rgba(105, 128, 160, 0.24)"


@st.cache_data(show_spinner=False)
def _load_latest_physics_dataset() -> pd.DataFrame | None:
    try:
        return load_pfr_dataset()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def _load_experimental_dataset() -> pd.DataFrame | None:
    try:
        return load_cv_reactor_experiment()
    except FileNotFoundError:
        return None
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def _validation_metric_snapshot(params_path: str | None) -> dict[str, float] | None:
    dataset = _load_experimental_dataset()
    if dataset is None:
        return None
    try:
        _, metrics = compare_experiment_to_surrogate(
            dataset,
            surrogate_params_path=params_path,
        )
    except Exception:  # noqa: BLE001
        return None
    return {
        "rmse_conversion": float(metrics["rmse_conversion"]),
        "mae_conversion": float(metrics["mae_conversion"]),
    }


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
                "methane_conversion_std": pred.get("methane_conversion_std", 0.0),
                "h2_yield_mol_per_mol_ch4": pred["h2_yield_mol_per_mol_ch4"],
                "h2_yield_std": pred.get("h2_yield_mol_per_mol_ch4_std", 0.0),
                "carbon_formation_index": pred["carbon_formation_index"],
                "carbon_formation_index_std": pred.get("carbon_formation_index_std", 0.0),
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


def _append_physics_label_to_dataset(
    label: dict[str, float], dataset_path: Path | None = None
) -> Path:
    if dataset_path is None:
        try:
            target_path = latest_physics_dataset(DEFAULT_PHYSICS_DIR)
        except FileNotFoundError:
            DEFAULT_PHYSICS_DIR.mkdir(parents=True, exist_ok=True)
            target_path = DEFAULT_PHYSICS_DIR / "physics_verified_labels.parquet"
    else:
        target_path = dataset_path

    row = pd.DataFrame([label])
    if target_path.exists():
        existing = pd.read_parquet(target_path)
        merged = pd.concat([existing, row], ignore_index=True)
    else:
        merged = row
    merged.to_parquet(target_path, index=False)
    return target_path


def _json_download_button(label: str, payload: dict[str, object], file_prefix: str) -> None:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    content = json.dumps(payload, indent=2)
    st.download_button(
        label=label,
        data=content.encode("utf-8"),
        file_name=f"{file_prefix}_{timestamp}.json",
        mime="application/json",
    )


def _pdf_summary_bytes(payload: dict[str, object]) -> bytes | None:
    try:
        fpdf_module = importlib.import_module("fpdf")
        fpdf_class = fpdf_module.FPDF
    except Exception:  # noqa: BLE001
        return None

    pdf = fpdf_class()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, "Synora Reactor Twin Summary", ln=1)
    pdf.set_font("Helvetica", size=8)
    for line in json.dumps(payload, indent=2).splitlines():
        pdf.multi_cell(0, 4, line)
    return bytes(pdf.output(dest="S"))


def _sync_control_from_slider(control_key: str) -> None:
    slider_key = f"{control_key}__slider"
    number_key = f"{control_key}__number"
    value = st.session_state[slider_key]
    st.session_state[control_key] = value
    st.session_state[number_key] = value


def _sync_control_from_number(control_key: str, cast_kind: str) -> None:
    number_key = f"{control_key}__number"
    slider_key = f"{control_key}__slider"
    value = st.session_state[number_key]
    normalized = int(value) if cast_kind == "int" else float(value)
    st.session_state[control_key] = normalized
    st.session_state[slider_key] = normalized


def _control_slider_with_stepper(
    *,
    label: str,
    control_key: str,
    min_value: int | float,
    max_value: int | float,
    step: int | float,
    cast_kind: str,
    number_format: str | None = None,
) -> None:
    slider_key = f"{control_key}__slider"
    number_key = f"{control_key}__number"
    value = st.session_state[control_key]
    st.session_state[slider_key] = value
    st.session_state[number_key] = value

    cols = st.columns([0.72, 0.28], gap="small")
    cols[0].slider(
        label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=slider_key,
        on_change=_sync_control_from_slider,
        args=(control_key,),
    )
    number_kwargs: dict[str, object] = {
        "label": f"{label} value",
        "min_value": min_value,
        "max_value": max_value,
        "step": step,
        "key": number_key,
        "label_visibility": "collapsed",
        "on_change": _sync_control_from_number,
        "args": (control_key, cast_kind),
    }
    if number_format is not None:
        number_kwargs["format"] = number_format
    cols[1].number_input(**number_kwargs)


def _hex_to_rgba(color: str, alpha: float) -> str:
    safe_alpha = max(0.0, min(1.0, alpha))
    value = color.lstrip("#")
    if len(value) != 6:
        return f"rgba(84, 173, 255, {safe_alpha:.3f})"
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {safe_alpha:.3f})"


def _trend_color(values: pd.Series) -> tuple[str, str]:
    series = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return ACCENT_GREEN, "▲"
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if end >= start:
        return ACCENT_GREEN, "▲"
    return ACCENT_RED, "▼"


def _sparkline_figure(values: pd.Series, *, color: str) -> tuple[go.Figure, str, str]:
    series = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        series = pd.Series([0.0, 0.0], dtype=float)
    elif len(series) == 1:
        series = pd.Series([float(series.iloc[0]), float(series.iloc[0])], dtype=float)

    trend_color, trend_symbol = _trend_color(series)
    point_count = len(series)
    if point_count <= 24:
        x_values = np.arange(point_count, dtype=float)
    else:
        x_values = np.linspace(0.0, 23.0, point_count, dtype=float)

    fig = go.Figure(
        go.Scatter(
            x=x_values,
            y=series.to_list(),
            mode="lines",
            line=dict(color=trend_color, width=2.2),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(trend_color, 0.18),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=56,
        margin=dict(l=2, r=2, t=2, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False, range=[0, 23], fixedrange=True)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    return fig, trend_color, trend_symbol


def _spark_metric_row(
    *,
    label: str,
    value: str,
    series: pd.Series,
    color: str,
) -> None:
    with st.container(border=True):
        spark_col, metric_col = st.columns([0.44, 0.56], gap="small")
        spark_fig, trend_color, trend_symbol = _sparkline_figure(series, color=color)
        spark_col.plotly_chart(
            spark_fig,
            width="stretch",
            config={"displayModeBar": False, "staticPlot": True},
        )
        metric_col.markdown(
            (
                "<div class='kpi-card-stack'>"
                f"<div class='kpi-card-label'>{label}</div>"
                f"<div class='kpi-card-value'>{value}</div>"
                f"<div class='kpi-card-trend' style='color:{trend_color}'>{trend_symbol}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _apply_consistent_grid(fig: go.Figure) -> None:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 16, 28, 0.74)",
        margin=dict(l=46, r=28, t=50, b=42),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor="rgba(118, 145, 178, 0.40)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor="rgba(118, 145, 178, 0.40)",
    )


def _render_design_explorer_hero(
    *,
    ranked: pd.DataFrame,
    validation_snapshot: dict[str, float] | None,
) -> None:
    if ranked.empty:
        x_values = np.array([1.0, 1.4, 1.8, 2.2, 2.7, 3.1, 3.6, 4.1, 4.5], dtype=float)
        annual_profit = np.array(
            [12200, 12850, 13640, 14520, 15280, 16100, 16880, 17520, 18240],
            dtype=float,
        )
        uncertainty = np.linspace(0.35, 0.95, len(x_values))
        hero_df = pd.DataFrame(
            {
                "fouling_risk_index": x_values,
                "annual_profit_usd": annual_profit,
                "uncertainty_proxy": uncertainty,
                "ood_score": np.linspace(2.6, 6.8, len(x_values)),
            }
        )
        candidate_df = pd.DataFrame(
            {
                "Design": [f"Design {chr(65 + idx)}" for idx in range(5)],
                "Fouling Index": np.round(x_values[:5], 2),
                "Profit ($/yr)": [f"${value:,.0f}" for value in annual_profit[:5]],
            }
        )
    else:
        hero_df = ranked.copy()
        profit_series = (
            hero_df["profit_per_hr"].astype(float)
            if "profit_per_hr" in hero_df.columns
            else pd.Series(np.zeros(len(hero_df)), index=hero_df.index, dtype=float)
        )
        fouling_series = (
            hero_df["fouling_risk_index"].astype(float)
            if "fouling_risk_index" in hero_df.columns
            else pd.Series(np.linspace(0.8, 3.5, len(hero_df)), index=hero_df.index, dtype=float)
        )
        conversion_std_series = (
            hero_df["conversion_std"].astype(float)
            if "conversion_std" in hero_df.columns
            else pd.Series(np.full(len(hero_df), 0.02), index=hero_df.index, dtype=float)
        )
        fouling_std_series = (
            hero_df["fouling_risk_index_std"].astype(float)
            if "fouling_risk_index_std" in hero_df.columns
            else pd.Series(np.full(len(hero_df), 0.03), index=hero_df.index, dtype=float)
        )
        ood_series = (
            hero_df["ood_score"].astype(float)
            if "ood_score" in hero_df.columns
            else pd.Series(np.full(len(hero_df), 2.0), index=hero_df.index, dtype=float)
        )
        hero_df["annual_profit_usd"] = profit_series * 8760.0
        hero_df["fouling_risk_index"] = fouling_series
        hero_df["ood_score"] = ood_series
        hero_df["uncertainty_proxy"] = np.clip(
            (conversion_std_series + fouling_std_series + (0.05 * ood_series)),
            0.0,
            1.0,
        )
        top_rows = hero_df.head(5).copy()
        candidate_df = pd.DataFrame(
            {
                "Design": [f"Design {chr(65 + idx)}" for idx in range(len(top_rows))],
                "Fouling Index": np.round(top_rows["fouling_risk_index"].astype(float), 2),
                "Profit ($/yr)": [
                    f"${value:,.0f}" for value in top_rows["annual_profit_usd"].astype(float)
                ],
            }
        )

    ordered = hero_df.sort_values("fouling_risk_index")
    ood_avg = float(hero_df["ood_score"].astype(float).mean())
    mae_value = (
        float(validation_snapshot["mae_conversion"])
        if validation_snapshot is not None
        else float(hero_df["uncertainty_proxy"].astype(float).mean())
    )
    bias_value = float(
        (
            hero_df["annual_profit_usd"].astype(float)
            - hero_df["annual_profit_usd"].astype(float).median()
        )
        .abs()
        .mean()
        / 1000.0
    )

    with st.container(border=True):
        st.markdown(
            "<div class='design-hero-header'>Pareto Frontier</div>"
            "<div class='design-hero-subheader'>Methane Reactor Design</div>",
            unsafe_allow_html=True,
        )
        hero_cols = st.columns([1.85, 1.0], gap="medium")

        hero_fig = go.Figure()
        hero_fig.add_trace(
            go.Scatter(
                x=hero_df["fouling_risk_index"],
                y=hero_df["annual_profit_usd"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=hero_df["uncertainty_proxy"],
                    colorscale="Turbo",
                    cmin=0.0,
                    cmax=1.0,
                    line=dict(width=0),
                    opacity=0.88,
                    showscale=False,
                ),
                hovertemplate=(
                    "Fouling: %{x:.2f}<br>"
                    "Annual Profit: $%{y:,.0f}<br>"
                    "Uncertainty: %{marker.color:.2f}<extra></extra>"
                ),
                name="Candidates",
            )
        )
        hero_fig.add_trace(
            go.Scatter(
                x=ordered["fouling_risk_index"],
                y=ordered["annual_profit_usd"],
                mode="lines+markers",
                line=dict(color="#77d5ff", width=2.6),
                marker=dict(size=6, color="#9ce8ff"),
                name="Pareto frontier",
                hoverinfo="skip",
            )
        )
        hero_fig.add_annotation(
            x=float(ordered["fouling_risk_index"].iloc[-1]),
            y=float(ordered["annual_profit_usd"].iloc[-1]),
            text="Pareto Frontier",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color="#dff1ff", size=11),
        )
        hero_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="Profit vs Fouling",
            height=470,
            margin=dict(l=42, r=18, t=50, b=40),
            paper_bgcolor="rgba(6,10,18,0.92)",
            plot_bgcolor="rgba(10,14,24,0.86)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        hero_fig.update_xaxes(
            title="Fouling Index",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
        )
        hero_fig.update_yaxes(
            title="Annual Profit (USD)",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
        )
        hero_cols[0].plotly_chart(
            hero_fig,
            width="stretch",
            config={"displayModeBar": False},
        )
        hero_cols[0].markdown(
            """
<div style="margin-top:0.35rem;">
  <div style="color:#9fb8d4;font-size:0.70rem;letter-spacing:0.06em;text-transform:uppercase;">
    Uncertainty Density
  </div>
  <div style="height:10px;border:1px solid rgba(88,120,158,0.55);background:linear-gradient(90deg,#12386f 0%,#2ea7ff 40%,#ffe38a 72%,#ff5a2f 100%);margin-top:0.24rem;"></div>
  <div style="display:flex;justify-content:space-between;color:#9bb5d2;font-size:0.66rem;letter-spacing:0.05em;margin-top:0.16rem;">
    <span>Low uncertainty</span><span>High uncertainty</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        hero_cols[1].markdown("#### Candidate Designs")
        hero_cols[1].dataframe(candidate_df, width="stretch", hide_index=True)
        kpi_cols = hero_cols[1].columns(3, gap="small")
        kpi_specs = [
            ("OOD AVG", f"{ood_avg:.2f}", "Out-of-distribution"),
            ("MAE", f"{mae_value:.2f}", "Prediction error"),
            ("BIAS", f"{bias_value:.2f}", "Profit bias ($k)"),
        ]
        for col, (title, value, subtitle) in zip(kpi_cols, kpi_specs, strict=True):
            with col.container(border=True):
                st.markdown(
                    (
                        f"<div class='design-hero-kpi-title'>{title}</div>"
                        f"<div class='design-hero-kpi-value'>{value}</div>"
                        f"<div style='color:#8ea8c6;font-size:0.66rem;letter-spacing:0.04em;'>{subtitle}</div>"
                    ),
                    unsafe_allow_html=True,
                )


_CONTROL_DEFAULTS: dict[str, int | float | bool] = {
    "control_hours": 72,
    "control_methane_kg_per_hr": 120.0,
    "control_temp_c": 980.0,
    "control_residence_time_s": 1.5,
    "control_maintenance_interval_hr": 72,
    "control_hydrogen_price": 4.7,
    "control_carbon_price": 0.22,
    "control_methane_price": 0.48,
    "control_variable_opex": 18.0,
    "control_fixed_opex": 10.0,
    "control_ticks_per_hour": 12,
    "control_zone_count": 3,
    "control_playback_ms": 180,
    "control_reaction_speed": 1,
    "control_reactor_debug": False,
    "control_thermal_view": True,
}
for control_key, default_value in _CONTROL_DEFAULTS.items():
    st.session_state.setdefault(control_key, default_value)

with st.sidebar:
    st.markdown("### Synora Control Room")
    st.caption("Primary controls are in the Reactor Overview tab.")
    st.write("Use this pane for future advanced diagnostics.")

st.session_state["control_thermal_view"] = True
st.session_state["control_reactor_debug"] = False

hours = int(st.session_state["control_hours"])
methane_kg_per_hr = float(st.session_state["control_methane_kg_per_hr"])
temp_c = float(st.session_state["control_temp_c"])
residence_time_s = float(st.session_state["control_residence_time_s"])
maintenance_interval_hr = int(st.session_state["control_maintenance_interval_hr"])
hydrogen_price = float(st.session_state["control_hydrogen_price"])
carbon_price = float(st.session_state["control_carbon_price"])
methane_price = float(st.session_state["control_methane_price"])
variable_opex = float(st.session_state["control_variable_opex"])
fixed_opex = float(st.session_state["control_fixed_opex"])

econ_inputs = EconInputs(
    methane_price_usd_per_kg=float(methane_price),
    hydrogen_price_usd_per_kg=float(hydrogen_price),
    carbon_price_usd_per_kg=float(carbon_price),
    variable_opex_usd_per_hr=float(variable_opex),
    fixed_opex_usd_per_hr=float(fixed_opex),
)

physics_df = _load_latest_physics_dataset()
experimental_df = _load_experimental_dataset()
params_path = _ensure_calibrated_params()
validation_snapshot = _validation_metric_snapshot(params_path)

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

st.title("Synora Industrial Digital Twin")
st.caption("Methane pyrolysis control room with simulation, optimization, and validation overlays.")

if physics_df is None:
    st.warning(
        "No physics parquet found in data/processed/physics_runs. "
        "Run scripts/physics/generate_pfr_dataset.py to enable physics overlays."
    )
elif params_path is None:
    st.warning("Physics data found, but surrogate params were not generated.")

tab_overview, tab_sensitivity, tab_envelope, tab_design, tab_architecture, tab_validation = st.tabs(
    [
        "Reactor Overview",
        "Twin Sensitivity",
        "Maintenance Envelope",
        "Design Explorer",
        "Reactor Architecture",
        "Experimental Validation",
    ]
)

with tab_overview:
    st.subheader("Reactor Control Room")
    layout_cols = st.columns([1.05, 2.4, 1.05], gap="medium")

    with layout_cols[0]:
        with st.expander("Process Controls", expanded=True):
            _control_slider_with_stepper(
                label="Simulation horizon (hours)",
                control_key="control_hours",
                min_value=12,
                max_value=336,
                step=1,
                cast_kind="int",
            )
            _control_slider_with_stepper(
                label="Methane feed (kg/hr)",
                control_key="control_methane_kg_per_hr",
                min_value=20.0,
                max_value=300.0,
                step=1.0,
                cast_kind="float",
                number_format="%.1f",
            )
            _control_slider_with_stepper(
                label="Reactor temperature (degC)",
                control_key="control_temp_c",
                min_value=850.0,
                max_value=1100.0,
                step=2.0,
                cast_kind="float",
                number_format="%.1f",
            )
            _control_slider_with_stepper(
                label="Residence time (s)",
                control_key="control_residence_time_s",
                min_value=0.1,
                max_value=5.0,
                step=0.05,
                cast_kind="float",
                number_format="%.2f",
            )
            _control_slider_with_stepper(
                label="Maintenance interval (hr)",
                control_key="control_maintenance_interval_hr",
                min_value=12,
                max_value=240,
                step=1,
                cast_kind="int",
            )
            _control_slider_with_stepper(
                label="Tick resolution (ticks/hr)",
                control_key="control_ticks_per_hour",
                min_value=2,
                max_value=24,
                step=1,
                cast_kind="int",
            )
            st.radio(
                "Thermal zone count",
                options=[2, 3],
                horizontal=True,
                key="control_zone_count",
            )
            _control_slider_with_stepper(
                label="Playback step (ms)",
                control_key="control_playback_ms",
                min_value=80,
                max_value=600,
                step=10,
                cast_kind="int",
            )
            _control_slider_with_stepper(
                label="Reaction Speed",
                control_key="control_reaction_speed",
                min_value=1,
                max_value=20,
                step=1,
                cast_kind="int",
            )
            st.caption("Reaction Speed changes physics timeline only.")
        with st.expander("Economics", expanded=False):
            st.number_input(
                "Hydrogen price (USD/kg)",
                min_value=0.0,
                step=0.1,
                key="control_hydrogen_price",
            )
            st.number_input(
                "Carbon price (USD/kg)",
                min_value=0.0,
                step=0.01,
                key="control_carbon_price",
            )
            st.number_input(
                "Methane price (USD/kg)",
                min_value=0.0,
                step=0.01,
                key="control_methane_price",
            )
            st.number_input(
                "Variable opex (USD/hr)",
                min_value=0.0,
                step=1.0,
                key="control_variable_opex",
            )
            st.number_input(
                "Fixed opex (USD/hr)",
                min_value=0.0,
                step=1.0,
                key="control_fixed_opex",
            )

    control_econ = EconInputs(
        methane_price_usd_per_kg=float(st.session_state["control_methane_price"]),
        hydrogen_price_usd_per_kg=float(st.session_state["control_hydrogen_price"]),
        carbon_price_usd_per_kg=float(st.session_state["control_carbon_price"]),
        variable_opex_usd_per_hr=float(st.session_state["control_variable_opex"]),
        fixed_opex_usd_per_hr=float(st.session_state["control_fixed_opex"]),
    )
    control_hours = int(st.session_state["control_hours"])
    control_zone_count = int(st.session_state["control_zone_count"])
    simulation_context = build_simulation_context(
        hours=control_hours,
        methane_kg_per_hr=float(st.session_state["control_methane_kg_per_hr"]),
        temp=float(st.session_state["control_temp_c"]),
        residence_time_s=float(st.session_state["control_residence_time_s"]),
        econ_inputs=control_econ,
        maintenance_interval_hr=int(st.session_state["control_maintenance_interval_hr"]),
        surrogate_params_path=params_path,
        ticks_per_hour=int(st.session_state["control_ticks_per_hour"]),
        zone_count=control_zone_count,
    )
    control_frames = simulation_context["frames"]
    if control_frames.empty:
        st.warning("Simulation context is empty. Adjust controls and rerun.")
    else:
        st.session_state.setdefault("overview_tick", 0)
        st.session_state.setdefault("overview_is_playing", False)
        st.session_state.setdefault("overview_last_advance_s", time.time())
        st.session_state.setdefault("overview_recenter_token", 0)
        max_tick = len(control_frames) - 1
        playback_step_s = max(0.08, float(st.session_state["control_playback_ms"]) / 1000.0)
        reaction_speed = max(1, int(st.session_state["control_reaction_speed"]))
        current_tick = int(np.clip(st.session_state["overview_tick"], 0, max_tick))
        now_s = time.time()

        with layout_cols[1]:
            toolbar = st.columns([0.20, 0.18, 0.18, 0.44])
            if toolbar[0].button(
                "Pause" if st.session_state["overview_is_playing"] else "Play",
                use_container_width=True,
            ):
                st.session_state["overview_is_playing"] = not st.session_state[
                    "overview_is_playing"
                ]
                st.session_state["overview_last_advance_s"] = now_s
            if toolbar[1].button("Reset", use_container_width=True):
                current_tick = 0
                st.session_state["overview_is_playing"] = False
                st.session_state["overview_last_advance_s"] = now_s
            if toolbar[2].button("Recenter", use_container_width=True):
                st.session_state["overview_recenter_token"] = (
                    int(st.session_state["overview_recenter_token"]) + 1
                )
            slider_tick = toolbar[3].slider(
                "Tick",
                min_value=0,
                max_value=max_tick,
                value=current_tick,
                label_visibility="collapsed",
            )
            if int(slider_tick) != current_tick:
                # Scrubber drag/release wins for this cycle; playback resumes from this tick.
                current_tick = int(slider_tick)
                st.session_state["overview_last_advance_s"] = now_s
            elif st.session_state["overview_is_playing"] and max_tick > 0:
                elapsed_s = max(0.0, now_s - float(st.session_state["overview_last_advance_s"]))
                steps = int(elapsed_s / playback_step_s)
                if steps > 0:
                    current_tick = (current_tick + (steps * reaction_speed)) % (max_tick + 1)
                    st.session_state["overview_last_advance_s"] = float(
                        st.session_state["overview_last_advance_s"]
                    ) + (steps * playback_step_s)

            st.session_state["overview_tick"] = int(np.clip(current_tick, 0, max_tick))
            frame = control_frames.iloc[int(st.session_state["overview_tick"])]
            visual_frame = build_visual_frame(frame, zone_count=control_zone_count)
            if validation_snapshot is not None:
                visual_frame["validation_rmse"] = float(validation_snapshot["rmse_conversion"])
                visual_frame["validation_mae"] = float(validation_snapshot["mae_conversion"])
            visual_frame["recenter_token"] = int(st.session_state["overview_recenter_token"])
            visual_frame["thermal_view"] = bool(st.session_state["control_thermal_view"])
            visual_frame["is_playing"] = bool(st.session_state["overview_is_playing"])
            render_reactor_3d(
                visual_frame,
                height=760,
                debug=False,
            )
            if st.session_state["overview_is_playing"] and max_tick > 0:
                elapsed_s = max(
                    0.0,
                    time.time() - float(st.session_state["overview_last_advance_s"]),
                )
                refresh_step_s = max(0.28, playback_step_s)
                wait_s = max(0.0, refresh_step_s - min(refresh_step_s, elapsed_s))
                if wait_s > 0:
                    time.sleep(wait_s)
                st.rerun()

        with layout_cols[2]:
            history = control_frames.iloc[: int(st.session_state["overview_tick"]) + 1]
            _spark_metric_row(
                label="Conversion",
                value=f"{100.0 * float(frame['conversion']):.1f}%",
                series=history["conversion"],
                color=ACCENT_BLUE,
            )
            _spark_metric_row(
                label="Reactor Health",
                value=f"{float(frame['health']):.3f}",
                series=history["health"],
                color=ACCENT_GREEN,
            )
            _spark_metric_row(
                label="Pressure Drop",
                value=f"{float(frame['pressure_drop_kpa']):.2f} kPa",
                series=history["pressure_drop_kpa"],
                color=ACCENT_ORANGE,
            )
            _spark_metric_row(
                label="Heater Power",
                value=f"{float(frame['heater_power_kw']):.1f} kW",
                series=history["heater_power_kw"],
                color=ACCENT_PURPLE,
            )
            _spark_metric_row(
                label="Fouling Index",
                value=f"{float(frame['fouling_index']):.3f}",
                series=history["fouling_index"],
                color="#f05454",
            )
            _spark_metric_row(
                label="Profit / hr",
                value=f"${float(frame['profit_per_hr']):.2f}",
                series=history["profit_per_hr"],
                color=ACCENT_GREEN,
            )
            _spark_metric_row(
                label="OOD",
                value="YES" if float(frame["is_out_of_distribution"]) > 0 else "NO",
                series=history["ood_score"],
                color="#ff7345",
            )
            _spark_metric_row(
                label="Confidence",
                value=f"{float(frame['confidence_index']):.2f}",
                series=history["confidence_index"],
                color=ACCENT_GREEN,
            )
            _spark_metric_row(
                label="H2 Fraction",
                value=f"{100.0 * float(visual_frame['hydrogen_fraction']):.1f}%",
                series=history["hydrogen_fraction"],
                color=ACCENT_BLUE,
            )

        chart_row_1 = st.columns(2)
        conversion_fig = go.Figure()
        conversion_fig.add_trace(
            go.Scatter(
                x=control_frames["time_hr"],
                y=control_frames["conversion"],
                mode="lines",
                line=dict(color=ACCENT_BLUE, width=2.6),
                name="Conversion",
            )
        )
        conversion_fig.add_trace(
            go.Scatter(
                x=[frame["time_hr"]],
                y=[frame["conversion"]],
                mode="markers",
                marker=dict(color=ACCENT_ORANGE, size=10, symbol="diamond"),
                name="Current tick",
            )
        )
        _apply_consistent_grid(conversion_fig)
        conversion_fig.update_layout(
            title="Conversion vs Time",
            xaxis_title="Time (hr)",
            yaxis_title="Methane conversion",
        )
        chart_row_1[0].plotly_chart(conversion_fig, width="stretch")

        dp_power_fig = go.Figure()
        dp_power_fig.add_trace(
            go.Scatter(
                x=control_frames["time_hr"],
                y=control_frames["pressure_drop_kpa"],
                mode="lines",
                line=dict(color=ACCENT_BLUE, width=2.2),
                name="DeltaP (kPa)",
            )
        )
        dp_power_fig.add_trace(
            go.Scatter(
                x=control_frames["time_hr"],
                y=control_frames["heater_power_kw"],
                mode="lines",
                line=dict(color=ACCENT_ORANGE, width=2.2),
                name="Heater power (kW)",
                yaxis="y2",
            )
        )
        _apply_consistent_grid(dp_power_fig)
        dp_power_fig.update_layout(
            title="DeltaP + Power Envelope",
            xaxis_title="Time (hr)",
            yaxis=dict(title="DeltaP (kPa)"),
            yaxis2=dict(
                title="Power (kW)",
                overlaying="y",
                side="right",
                showgrid=False,
                linecolor="rgba(118, 145, 178, 0.40)",
            ),
        )
        chart_row_1[1].plotly_chart(dp_power_fig, width="stretch")

        chart_row_2 = st.columns(2)
        quality_fig = go.Figure()
        quality_fig.add_trace(
            go.Scatter(
                x=control_frames["time_hr"],
                y=control_frames["carbon_formation_index"],
                mode="lines",
                line=dict(color=ACCENT_ORANGE, width=2.2),
                name="Carbon quality index",
            )
        )
        quality_fig.add_trace(
            go.Scatter(
                x=control_frames["time_hr"],
                y=control_frames["confidence_index"],
                mode="lines",
                line=dict(color=ACCENT_GREEN, width=2.0, dash="dot"),
                name="Confidence index",
                yaxis="y2",
            )
        )
        _apply_consistent_grid(quality_fig)
        quality_fig.update_layout(
            title="Carbon Quality Index / Confidence",
            xaxis_title="Time (hr)",
            yaxis=dict(title="Carbon index"),
            yaxis2=dict(
                title="Confidence",
                overlaying="y",
                side="right",
                showgrid=False,
                linecolor="rgba(118, 145, 178, 0.40)",
            ),
        )
        chart_row_2[0].plotly_chart(quality_fig, width="stretch")

        envelope_df = _profit_envelope(
            intervals_hr=tuple(range(12, 241, 12)),
            hours=int(control_hours),
            methane_kg_per_hr=float(st.session_state["control_methane_kg_per_hr"]),
            temp_c=float(st.session_state["control_temp_c"]),
            residence_time_s=float(st.session_state["control_residence_time_s"]),
            econ_inputs=control_econ,
            params_path=params_path,
        )
        profit_fig = go.Figure()
        profit_fig.add_trace(
            go.Scatter(
                x=envelope_df["maintenance_interval_hr"],
                y=envelope_df["cum_profit_usd"],
                mode="lines+markers",
                marker=dict(size=7),
                line=dict(color=ACCENT_GREEN, width=2.4),
                name="Cumulative profit",
            )
        )
        _apply_consistent_grid(profit_fig)
        profit_fig.update_layout(
            title="Profit vs Maintenance",
            xaxis_title="Maintenance interval (hr)",
            yaxis_title="Cumulative profit (USD)",
        )
        chart_row_2[1].plotly_chart(profit_fig, width="stretch")

        twin_payload = {
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "synora_version": SYNORA_VERSION,
            "controls": {
                "hours": control_hours,
                "methane_kg_per_hr": float(st.session_state["control_methane_kg_per_hr"]),
                "temp_c": float(st.session_state["control_temp_c"]),
                "residence_time_s": float(st.session_state["control_residence_time_s"]),
                "maintenance_interval_hr": int(st.session_state["control_maintenance_interval_hr"]),
                "ticks_per_hour": int(st.session_state["control_ticks_per_hour"]),
                "zone_count": control_zone_count,
            },
            "latest_frame": {key: float(value) for key, value in frame.items()},
        }
        download_cols = st.columns(2)
        with download_cols[0]:
            _json_download_button("Download Twin JSON", twin_payload, "reactor_twin_report")
        with download_cols[1]:
            pdf_bytes = _pdf_summary_bytes(twin_payload)
            if pdf_bytes is not None:
                timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
                st.download_button(
                    "Download Twin PDF",
                    data=pdf_bytes,
                    file_name=f"reactor_twin_summary_{timestamp}.pdf",
                    mime="application/pdf",
                )
            else:
                st.caption("PDF summary export is optional; install `fpdf2` to enable.")

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
    hero_placeholder = st.empty()

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
        ranked_for_hero = pd.DataFrame()
    elif {"constraint_violation_count", "score"}.issubset(leaderboard.columns):
        ranked_for_hero = leaderboard.sort_values(
            ["constraint_violation_count", "score"], ascending=[True, False]
        )
    elif "score" in leaderboard.columns:
        ranked_for_hero = leaderboard.sort_values("score", ascending=False)
    else:
        ranked_for_hero = leaderboard.copy()
    with hero_placeholder.container():
        _render_design_explorer_hero(
            ranked=ranked_for_hero, validation_snapshot=validation_snapshot
        )

    if leaderboard.empty:
        st.info("Run optimization to generate candidate reactor designs.")
    else:
        display_columns = [
            "score",
            "profit_per_hr",
            "conversion",
            "conversion_std",
            "h2_rate",
            "fouling_risk_index",
            "fouling_risk_index_std",
            "ood_score",
            "is_out_of_distribution",
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
        ranked = ranked_for_hero.copy()

        st.markdown("**Leaderboard**")
        st.dataframe(ranked[available_columns], width="stretch")

        pareto_cols = st.columns(2)
        ood_mask = ranked["is_out_of_distribution"].astype(bool)
        marker_colors = np.where(ood_mask, "#ff4d4f", ACCENT_BLUE)
        marker_symbols = np.where(ood_mask, "x", "circle")

        pareto_fig = go.Figure()
        pareto_fig.add_trace(
            go.Scatter(
                x=ranked["fouling_risk_index"],
                y=ranked["profit_per_hr"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=marker_colors,
                    symbol=marker_symbols,
                ),
                text=[f"Design {idx}" for idx in ranked.index],
                hovertemplate=(
                    "Fouling risk: %{x:.3f}<br>"
                    "Profit/hr: %{y:.2f}<br>"
                    "OOD flag: %{marker.symbol}<extra></extra>"
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
                marker=dict(size=9, color=marker_colors, symbol=marker_symbols),
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

        confidence_label = "LOW"
        if bool(selected_metrics["is_out_of_distribution"]):
            confidence_label = "LOW (OOD)"
        elif (
            float(selected_metrics["conversion_std"]) < 0.01
            and float(selected_metrics["h2_yield_std"]) < 0.03
        ):
            confidence_label = "HIGH"
        elif float(selected_metrics["conversion_std"]) < 0.03:
            confidence_label = "MEDIUM"

        detail_cols = st.columns(4)
        detail_cols[0].metric(
            "Selected Profit/hr (USD)", f"{float(selected_metrics['profit_per_hr']):.2f}"
        )
        detail_cols[1].metric("Selected Conversion", f"{float(selected_metrics['conversion']):.3f}")
        detail_cols[2].metric(
            "Fouling Risk Index", f"{float(selected_metrics['fouling_risk_index']):.3f}"
        )
        detail_cols[3].metric("Residence Time (s)", f"{selected_design.residence_time_s:.2f}")
        st.markdown(
            f"**Confidence:** `{confidence_label}` | **OOD score:** `{float(selected_metrics['ood_score']):.3f}`"
        )

        uncertainty_fig = go.Figure()
        x_labels = ["Conversion", "H2 Yield", "Fouling Risk"]
        surrogate_means = [
            float(selected_metrics["conversion"]),
            float(selected_metrics["h2_yield_mol_per_mol_ch4"]),
            float(selected_metrics["fouling_risk_index"]),
        ]
        surrogate_err = [
            2.0 * float(selected_metrics["conversion_std"]),
            2.0 * float(selected_metrics["h2_yield_std"]),
            2.0 * float(selected_metrics["fouling_risk_index_std"]),
        ]
        uncertainty_fig.add_trace(
            go.Bar(
                x=x_labels,
                y=surrogate_means,
                error_y=dict(type="data", array=surrogate_err, visible=True),
                marker_color=[ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE],
                name="Surrogate mean +/- 2sigma",
            )
        )

        physics_label = st.session_state.get("physics_verified_label")
        if isinstance(physics_label, dict):
            physics_fouling = float(physics_label["carbon_formation_index"]) * (
                1.0 - selected_design.carbon_removal_eff
            )
            uncertainty_fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=[
                        float(physics_label["methane_conversion"]),
                        float(physics_label["h2_yield_mol_per_mol_ch4"]),
                        physics_fouling,
                    ],
                    mode="markers",
                    marker=dict(color="#ffffff", size=10, symbol="diamond"),
                    name="Physics verification",
                )
            )
        uncertainty_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="Selected Design: Surrogate Uncertainty vs Physics",
            yaxis_title="Metric value",
        )
        st.plotly_chart(uncertainty_fig, width="stretch")

        verify_cols = st.columns(3)
        append_verify = verify_cols[0].checkbox(
            "Append verification to dataset",
            value=True,
            key="append_verify_toggle",
        )
        refit_after_append = verify_cols[1].checkbox(
            "Refit surrogate after append",
            value=True,
            disabled=not append_verify,
            key="refit_after_append_toggle",
        )
        run_verify = verify_cols[2].button("Physics Verify Selected Design", type="secondary")

        if run_verify:
            with st.spinner("Running Cantera physics verification..."):
                try:
                    label = label_pfr_case(
                        temperature_c=selected_design.temp_c,
                        residence_time_s=selected_design.residence_time_s,
                        pressure_atm=selected_design.pressure_atm,
                        dilution_frac=selected_design.dilution_frac,
                        methane_kg_per_hr=selected_design.methane_kg_per_hr,
                    )
                    st.session_state["physics_verified_label"] = label
                    st.success("Physics verification complete.")
                    if append_verify:
                        appended_path = _append_physics_label_to_dataset(label)
                        st.success(f"Appended verified point to `{appended_path}`.")
                        if refit_after_append:
                            target_params = (
                                DEFAULT_PARAMS_PATH if params_path is None else Path(params_path)
                            )
                            calibrate_and_store(
                                dataset_path=appended_path,
                                params_path=target_params,
                                ensemble_size=7,
                                random_seed=42,
                                verbose=False,
                            )
                            st.success("Surrogate refit completed from updated dataset.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Physics verification failed: {exc}")

        report_cols = st.columns(2)
        uncertainty_payload = {
            "conversion_ci_lower": float(selected_metrics["conversion_ci_lower"]),
            "conversion_ci_upper": float(selected_metrics["conversion_ci_upper"]),
            "h2_yield_ci_lower": float(selected_metrics["h2_yield_ci_lower"]),
            "h2_yield_ci_upper": float(selected_metrics["h2_yield_ci_upper"]),
            "fouling_risk_ci_lower": float(selected_metrics["fouling_risk_ci_lower"]),
            "fouling_risk_ci_upper": float(selected_metrics["fouling_risk_ci_upper"]),
        }
        design_report_payload = {
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "report_type": "design_explorer",
            "synora_version": SYNORA_VERSION,
            "design": selected_design.to_dict(),
            "metrics": dict(selected_metrics),
            "uncertainty": uncertainty_payload,
            "ood_score": float(selected_metrics["ood_score"]),
        }
        with report_cols[0]:
            _json_download_button("Download Design Report", design_report_payload, "design_report")
        with report_cols[1]:
            pdf_bytes = _pdf_summary_bytes(design_report_payload)
            if pdf_bytes is not None:
                timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
                st.download_button(
                    "Download Design PDF",
                    data=pdf_bytes,
                    file_name=f"design_report_{timestamp}.pdf",
                    mime="application/pdf",
                )
            else:
                st.markdown(
                    "Optional PDF export unavailable. Install `fpdf2` to enable downloadable summaries."
                )

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

with tab_architecture:
    st.subheader("Reactor Architecture")
    st.caption(
        "CFD-lite multi-zone architecture (2-3 zones) with thermal and pressure-drop constraints."
    )

    mode = st.radio(
        "Architecture mode",
        options=["Single-Zone", "Multi-Zone"],
        horizontal=True,
        key="arch_mode_toggle",
    )

    global_cols = st.columns(5)
    arch_methane_kg_per_hr = global_cols[0].slider(
        "Architecture methane (kg/hr)",
        min_value=20,
        max_value=300,
        value=110,
        step=5,
        key="arch_methane_kg_per_hr",
    )
    arch_pressure_atm = global_cols[1].slider(
        "Architecture pressure (atm)",
        min_value=1.0,
        max_value=3.0,
        value=1.3,
        step=0.1,
        key="arch_pressure_atm",
    )
    arch_dilution = global_cols[2].slider(
        "Architecture dilution frac",
        min_value=0.55,
        max_value=0.92,
        value=0.80,
        step=0.01,
        key="arch_dilution_frac",
    )
    arch_carbon_removal = global_cols[3].slider(
        "Carbon removal eff",
        min_value=0.0,
        max_value=0.95,
        value=0.55,
        step=0.01,
        key="arch_carbon_removal_eff",
    )
    arch_ambient_temp_c = global_cols[4].slider(
        "Ambient temp (degC)",
        min_value=0,
        max_value=100,
        value=25,
        step=1,
        key="arch_ambient_temp_c",
    )

    constraint_cols = st.columns(3)
    arch_material_tmax_c = constraint_cols[0].slider(
        "Material Tmax (degC)",
        min_value=900,
        max_value=1400,
        value=1120,
        step=10,
        key="arch_material_tmax_c",
    )
    arch_dp_max_kpa = constraint_cols[1].slider(
        "DeltaP max (kPa)",
        min_value=5,
        max_value=120,
        value=35,
        step=1,
        key="arch_dp_max_kpa",
    )
    arch_power_max_kw = constraint_cols[2].slider(
        "Power max (kW)",
        min_value=50,
        max_value=5000,
        value=1400,
        step=25,
        key="arch_power_max_kw",
    )

    architecture_metrics: dict[str, float | bool | list[str]]
    architecture_design: ReactorDesign | MultiZoneDesign
    architecture_zone_temps: list[float]
    architecture_zone_taus: list[float]

    if mode == "Single-Zone":
        single_cols = st.columns(3)
        single_temp_c = single_cols[0].slider(
            "Zone temp (degC)",
            min_value=850,
            max_value=1100,
            value=980,
            step=10,
            key="arch_single_temp_c",
        )
        single_length_m = single_cols[1].slider(
            "Zone length (m)",
            min_value=0.5,
            max_value=3.0,
            value=1.4,
            step=0.1,
            key="arch_single_length_m",
        )
        single_diameter_m = single_cols[2].slider(
            "Zone diameter (m)",
            min_value=0.04,
            max_value=0.20,
            value=0.09,
            step=0.005,
            key="arch_single_diameter_m",
        )

        architecture_design = ReactorDesign(
            length_m=float(single_length_m),
            diameter_m=float(single_diameter_m),
            pressure_atm=float(arch_pressure_atm),
            temp_c=float(single_temp_c),
            methane_kg_per_hr=float(arch_methane_kg_per_hr),
            dilution_frac=float(arch_dilution),
            carbon_removal_eff=float(arch_carbon_removal),
        )
        architecture_metrics = evaluate_design_surrogate(
            architecture_design,
            surrogate_params_path=params_path,
            econ_inputs=econ_inputs,
        )
        architecture_zone_temps = [float(single_temp_c)]
        architecture_zone_taus = [float(architecture_design.residence_time_s)]
    else:
        zone_count = st.slider(
            "Zone count",
            min_value=2,
            max_value=3,
            value=2,
            step=1,
            key="arch_zone_count",
        )
        default_diameter_m = st.slider(
            "Default diameter fallback (m)",
            min_value=0.04,
            max_value=0.20,
            value=0.09,
            step=0.005,
            key="arch_default_diameter_m",
        )
        zones: list[ZoneDesign] = []
        for idx in range(zone_count):
            zone_cols = st.columns(4)
            zone_temp = zone_cols[0].slider(
                f"Zone {idx + 1} temp (degC)",
                min_value=850,
                max_value=1100,
                value=930 + (idx * 40),
                step=10,
                key=f"arch_zone_{idx + 1}_temp",
            )
            zone_length = zone_cols[1].slider(
                f"Zone {idx + 1} length (m)",
                min_value=0.20,
                max_value=2.00,
                value=0.90,
                step=0.05,
                key=f"arch_zone_{idx + 1}_length",
            )
            zone_diameter = zone_cols[2].slider(
                f"Zone {idx + 1} diameter (m)",
                min_value=0.04,
                max_value=0.20,
                value=0.09,
                step=0.005,
                key=f"arch_zone_{idx + 1}_diameter",
            )
            zone_insulation = zone_cols[3].slider(
                f"Zone {idx + 1} insulation factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                key=f"arch_zone_{idx + 1}_insulation",
            )
            zones.append(
                ZoneDesign(
                    temp_c=float(zone_temp),
                    length_m=float(zone_length),
                    diameter_m=float(zone_diameter),
                    insulation_factor=float(zone_insulation),
                )
            )

        architecture_design = MultiZoneDesign(
            zones=zones,
            methane_kg_per_hr=float(arch_methane_kg_per_hr),
            pressure_atm=float(arch_pressure_atm),
            dilution_frac=float(arch_dilution),
            carbon_removal_eff=float(arch_carbon_removal),
            ambient_temp_c=float(arch_ambient_temp_c),
            material_tmax_c=float(arch_material_tmax_c),
            dp_max_kpa=float(arch_dp_max_kpa),
            power_max_kw=float(arch_power_max_kw),
            default_diameter_m=float(default_diameter_m),
        )
        architecture_metrics = evaluate_multizone_surrogate(
            architecture_design,
            surrogate_params_path=params_path,
            econ_inputs=econ_inputs,
        )
        architecture_zone_temps = architecture_design.zone_temperatures_c
        architecture_zone_taus = architecture_design.zone_residence_time_s

    kpi_cols = st.columns(8)
    is_ood_flag = bool(architecture_metrics.get("is_out_of_distribution", False))
    ood_score = float(architecture_metrics.get("ood_score", 0.0))
    dp_total_kpa = float(
        architecture_metrics.get(
            "dp_total_kpa",
            float(architecture_metrics["pressure_drop_proxy"]) * float(arch_dp_max_kpa),
        )
    )
    q_loss_kw = float(
        architecture_metrics.get(
            "q_loss_kw",
            float(architecture_metrics["heat_loss_proxy"]) * float(arch_power_max_kw),
        )
    )
    q_required_kw = float(architecture_metrics.get("q_required_kw", q_loss_kw))
    confidence_index = max(
        0.0,
        1.0
        - min(
            1.0,
            (2.0 * float(architecture_metrics.get("conversion_std", 0.0))) + (0.25 * ood_score),
        ),
    )
    kpi_cols[0].metric("Conversion", f"{float(architecture_metrics['conversion']):.3f}")
    kpi_cols[1].metric("H2 rate (kg/hr)", f"{float(architecture_metrics['h2_rate']):.2f}")
    kpi_cols[2].metric("DeltaP total (kPa)", f"{dp_total_kpa:.2f}")
    kpi_cols[3].metric("Q required (kW)", f"{q_required_kw:.1f}")
    kpi_cols[4].metric("Q loss (kW)", f"{q_loss_kw:.1f}")
    kpi_cols[5].metric("Fouling risk", f"{float(architecture_metrics['fouling_risk_index']):.3f}")
    kpi_cols[6].metric("OOD", "YES" if is_ood_flag else "NO")
    kpi_cols[7].metric("Confidence", f"{confidence_index:.2f}")

    profile_cols = st.columns(2)
    zone_idx = list(range(1, len(architecture_zone_temps) + 1))
    profile_fig = go.Figure()
    profile_fig.add_trace(
        go.Scatter(
            x=zone_idx,
            y=architecture_zone_temps,
            mode="lines+markers",
            marker=dict(color=ACCENT_ORANGE, size=9),
            line=dict(color=ACCENT_ORANGE, width=2.4),
            name="Zone temperature",
        )
    )
    profile_fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Zone Temperature Profile",
        xaxis_title="Zone index",
        yaxis_title="Temperature (degC)",
    )
    profile_cols[0].plotly_chart(profile_fig, width="stretch")

    zone_conv_means: list[float] = []
    zone_conv_stds: list[float] = []
    for idx in range(len(architecture_zone_temps)):
        if mode == "Single-Zone":
            zone_conv_means.append(float(architecture_metrics["conversion"]))
            zone_conv_stds.append(float(architecture_metrics.get("conversion_std", 0.0)))
        else:
            zone_conv_means.append(
                float(architecture_metrics.get(f"zone_{idx + 1}_conv_mean", 0.0))
            )
            zone_conv_stds.append(float(architecture_metrics.get(f"zone_{idx + 1}_conv_std", 0.0)))

    conversion_fig = go.Figure()
    conversion_fig.add_trace(
        go.Bar(
            x=zone_idx,
            y=zone_conv_means,
            error_y=dict(
                type="data", array=[2.0 * value for value in zone_conv_stds], visible=True
            ),
            marker_color=ACCENT_BLUE,
            name="Surrogate mean +/- 2sigma",
        )
    )

    physics_overlay = st.session_state.get("architecture_physics_labels")
    if isinstance(physics_overlay, list) and physics_overlay:
        conversion_fig.add_trace(
            go.Scatter(
                x=[int(label["zone_index"]) for label in physics_overlay],
                y=[float(label["methane_conversion"]) for label in physics_overlay],
                mode="markers",
                marker=dict(color="#ffffff", size=10, symbol="diamond"),
                name="Physics verify",
            )
        )
    conversion_fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Zone Conversion Prediction with Uncertainty",
        xaxis_title="Zone index",
        yaxis_title="Methane conversion",
    )
    profile_cols[1].plotly_chart(conversion_fig, width="stretch")

    st.markdown(
        f"**Zone residence times (s):** `{', '.join(f'{tau:.3f}' for tau in architecture_zone_taus)}`"
    )
    st.markdown(
        "**Constraint violations:** "
        + ", ".join(architecture_metrics.get("constraint_violations", []))
        if architecture_metrics.get("constraint_violations")
        else "**Constraint violations:** none"
    )

    verify_cols = st.columns(3)
    append_arch_verify = verify_cols[0].checkbox(
        "Append architecture verification labels",
        value=True,
        key="arch_append_verify_toggle",
    )
    refit_arch = verify_cols[1].checkbox(
        "Refit surrogate after append",
        value=True,
        disabled=not append_arch_verify,
        key="arch_refit_verify_toggle",
    )
    run_arch_verify = verify_cols[2].button(
        "Physics Verify Architecture",
        key="run_architecture_verify_button",
    )

    if run_arch_verify:
        with st.spinner("Running physics verification for architecture..."):
            try:
                labels: list[dict[str, float | int]] = []
                if isinstance(architecture_design, MultiZoneDesign):
                    for idx, zone in enumerate(architecture_design.zones):
                        label = label_pfr_case(
                            temperature_c=zone.temp_c,
                            residence_time_s=architecture_design.zone_residence_time_s[idx],
                            pressure_atm=architecture_design.zone_pressures_atm[idx],
                            dilution_frac=architecture_design.dilution_frac,
                            methane_kg_per_hr=architecture_design.methane_kg_per_hr,
                        )
                        label["zone_index"] = idx + 1
                        labels.append(label)
                else:
                    label = label_pfr_case(
                        temperature_c=architecture_design.temp_c,
                        residence_time_s=architecture_design.residence_time_s,
                        pressure_atm=architecture_design.pressure_atm,
                        dilution_frac=architecture_design.dilution_frac,
                        methane_kg_per_hr=architecture_design.methane_kg_per_hr,
                    )
                    label["zone_index"] = 1
                    labels.append(label)

                st.session_state["architecture_physics_labels"] = labels
                st.success("Physics verification complete.")

                if append_arch_verify:
                    appended_path: Path | None = None
                    for label in labels:
                        row = dict(label)
                        row.pop("zone_index", None)
                        appended_path = _append_physics_label_to_dataset(row)
                    if appended_path is not None:
                        st.success(f"Appended verification labels to `{appended_path}`.")
                        if refit_arch:
                            target_params = (
                                DEFAULT_PARAMS_PATH if params_path is None else Path(params_path)
                            )
                            calibrate_and_store(
                                dataset_path=appended_path,
                                params_path=target_params,
                                ensemble_size=7,
                                random_seed=42,
                                verbose=False,
                            )
                            st.success("Surrogate refit completed from architecture labels.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Architecture verification failed: {exc}")

    report_cols = st.columns(2)
    architecture_uncertainty = {
        "conversion_ci_lower": float(architecture_metrics["conversion_ci_lower"]),
        "conversion_ci_upper": float(architecture_metrics["conversion_ci_upper"]),
        "h2_yield_ci_lower": float(architecture_metrics["h2_yield_ci_lower"]),
        "h2_yield_ci_upper": float(architecture_metrics["h2_yield_ci_upper"]),
        "fouling_risk_ci_lower": float(architecture_metrics["fouling_risk_ci_lower"]),
        "fouling_risk_ci_upper": float(architecture_metrics["fouling_risk_ci_upper"]),
    }
    architecture_payload = {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "report_type": "reactor_architecture",
        "synora_version": SYNORA_VERSION,
        "design": architecture_design.to_dict() if hasattr(architecture_design, "to_dict") else {},
        "metrics": dict(architecture_metrics),
        "uncertainty": architecture_uncertainty,
        "ood_score": float(architecture_metrics.get("ood_score", 0.0)),
    }
    with report_cols[0]:
        _json_download_button(
            "Download Architecture Report",
            architecture_payload,
            "architecture_report",
        )
    with report_cols[1]:
        pdf_bytes = _pdf_summary_bytes(architecture_payload)
        if pdf_bytes is not None:
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            st.download_button(
                "Download Architecture PDF",
                data=pdf_bytes,
                file_name=f"architecture_report_{timestamp}.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("Optional PDF summary unavailable. Install `fpdf2` to enable.")

    if mode == "Multi-Zone":
        st.divider()
        st.markdown("**Multi-Zone Optimization Explorer**")
        opt_cols = st.columns(4)
        mz_top_k = opt_cols[0].slider(
            "Top K",
            min_value=5,
            max_value=25,
            value=10,
            step=1,
            key="arch_mz_top_k",
        )
        mz_generations = opt_cols[1].slider(
            "Generations",
            min_value=4,
            max_value=20,
            value=8,
            step=1,
            key="arch_mz_generations",
        )
        mz_population = opt_cols[2].slider(
            "Population",
            min_value=40,
            max_value=240,
            value=100,
            step=10,
            key="arch_mz_population",
        )
        mz_seed = opt_cols[3].number_input(
            "Seed",
            min_value=1,
            max_value=100000,
            value=17,
            step=1,
            key="arch_mz_seed",
        )

        if st.button("Run Multi-Zone Optimization", key="run_multizone_optimization_button"):
            mz_bounds = MultiZoneBounds(
                ambient_temp_c=float(arch_ambient_temp_c),
                material_tmax_c=float(arch_material_tmax_c),
                dp_max_kpa=float(arch_dp_max_kpa),
                power_max_kw=float(arch_power_max_kw),
            )
            evaluations = propose_multizone_designs(
                top_k=int(mz_top_k),
                zones=int(zone_count),
                generations=int(mz_generations),
                population_size=int(mz_population),
                seed=int(mz_seed),
                bounds=mz_bounds,
                surrogate_params_path=params_path,
                econ_inputs=econ_inputs,
            )
            st.session_state["architecture_multizone_leaderboard"] = multizone_evaluations_to_frame(
                evaluations
            )

        multizone_leaderboard = st.session_state.get(
            "architecture_multizone_leaderboard", pd.DataFrame()
        )
        if not multizone_leaderboard.empty:
            ranked_mz = multizone_leaderboard.sort_values(
                ["constraint_violation_count", "score"], ascending=[True, False]
            )
            show_cols = [
                "score",
                "profit_per_hr",
                "conversion",
                "conversion_std",
                "h2_rate",
                "fouling_risk_index",
                "dp_total_kpa",
                "q_required_kw",
                "ood_score",
                "constraint_violation_count",
                "zone_1_temp_c",
                "zone_2_temp_c",
                "zone_3_temp_c",
                "zone_1_tau_s",
                "zone_2_tau_s",
                "zone_3_tau_s",
            ]
            visible_cols = [column for column in show_cols if column in ranked_mz.columns]
            st.dataframe(ranked_mz[visible_cols], width="stretch")

            ood_mask = ranked_mz["is_out_of_distribution"].astype(bool)
            pareto_fig = go.Figure()
            pareto_fig.add_trace(
                go.Scatter(
                    x=ranked_mz["fouling_risk_index"],
                    y=ranked_mz["profit_per_hr"],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=np.where(ood_mask, "#ff4d4f", ACCENT_GREEN),
                        symbol=np.where(ood_mask, "x", "circle"),
                    ),
                    name="Candidates",
                    hovertemplate="Fouling: %{x:.3f}<br>Profit/hr: %{y:.2f}<extra></extra>",
                )
            )
            pareto_fig.update_layout(
                template=PLOT_TEMPLATE,
                title="Multi-Zone Pareto: Profit vs Fouling",
                xaxis_title="Fouling risk index",
                yaxis_title="Profit per hour (USD)",
            )
            st.plotly_chart(pareto_fig, width="stretch")

with tab_validation:
    st.subheader("Experimental Validation")
    st.caption(
        "Investor-grade overlay: experimental methane conversion versus surrogate mean with uncertainty."
    )

    if experimental_df is None:
        st.warning(
            "No experimental parquet found at "
            f"`{DEFAULT_EXPERIMENTAL_DATASET_PATH}`. Validation tab is unavailable."
        )
    else:
        overlay_df, validation_metrics = compare_experiment_to_surrogate(
            experimental_df,
            surrogate_params_path=params_path,
        )

        kpi_cols = st.columns(6)
        kpi_cols[0].metric(
            "RMSE (conversion)", f"{float(validation_metrics['rmse_conversion']):.4f}"
        )
        kpi_cols[1].metric("MAE (conversion)", f"{float(validation_metrics['mae_conversion']):.4f}")
        kpi_cols[2].metric(
            "Bias (conversion)", f"{float(validation_metrics['bias_conversion']):.4f}"
        )
        kpi_cols[3].metric("OOD count", f"{int(validation_metrics['ood_count'])}")
        kpi_cols[4].metric("OOD fraction", f"{float(validation_metrics['ood_fraction']):.2%}")
        kpi_cols[5].metric("Samples", f"{int(validation_metrics['n_samples'])}")

        dataset_path_label = str(
            experimental_df.attrs.get("dataset_path", DEFAULT_EXPERIMENTAL_DATASET_PATH)
        )
        st.markdown(f"**Dataset:** `{dataset_path_label}`")
        st.markdown(f"**Validation-axis note:** {VALIDATION_AXIS_NOTE}")

        chart_cols = st.columns(2)

        conversion_fig = go.Figure()
        conversion_fig.add_trace(
            go.Scatter(
                x=overlay_df["time_s"],
                y=overlay_df["methane_conversion_exp"],
                mode="lines+markers",
                name="Experiment conversion",
                line=dict(color=ACCENT_ORANGE, width=2.2),
                marker=dict(size=6),
            )
        )
        conversion_fig.add_trace(
            go.Scatter(
                x=overlay_df["time_s"],
                y=overlay_df["methane_conversion_pred_mean"],
                mode="lines",
                name="Surrogate mean",
                line=dict(color=ACCENT_BLUE, width=2.4),
            )
        )
        conversion_fig.add_trace(
            go.Scatter(
                x=overlay_df["time_s"],
                y=overlay_df["pred_ci_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        conversion_fig.add_trace(
            go.Scatter(
                x=overlay_df["time_s"],
                y=overlay_df["pred_ci_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(76,141,255,0.18)",
                name="Prediction +/- 2sigma",
                hoverinfo="skip",
            )
        )

        ood_rows = overlay_df[overlay_df["is_out_of_distribution"].astype(bool)]
        if not ood_rows.empty:
            conversion_fig.add_trace(
                go.Scatter(
                    x=ood_rows["time_s"],
                    y=ood_rows["methane_conversion_exp"],
                    mode="markers",
                    marker=dict(color="#ff4d4f", size=9, symbol="x"),
                    name="OOD flagged samples",
                )
            )
        conversion_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="Methane Conversion: Experiment vs Surrogate",
            xaxis_title="Time (s) used as tau proxy",
            yaxis_title="Conversion",
        )
        chart_cols[0].plotly_chart(conversion_fig, width="stretch")

        hydrogen_fig = go.Figure()
        hydrogen_fig.add_trace(
            go.Scatter(
                x=overlay_df["time_s"],
                y=overlay_df["hydrogen_mol_percent_exp"],
                mode="lines+markers",
                name="Experiment H2 mol%",
                line=dict(color=ACCENT_GREEN, width=2.2),
                marker=dict(size=6),
            )
        )

        has_h2_prediction = bool(
            overlay_df["hydrogen_prediction_available"].astype(bool).any()
        ) and bool(overlay_df["hydrogen_mol_percent_pred_mean"].notna().any())
        if has_h2_prediction:
            hydrogen_fig.add_trace(
                go.Scatter(
                    x=overlay_df["time_s"],
                    y=overlay_df["hydrogen_mol_percent_pred_mean"],
                    mode="lines",
                    name="Surrogate H2 mean",
                    line=dict(color=ACCENT_BLUE, width=2.3),
                )
            )
            hydrogen_fig.add_trace(
                go.Scatter(
                    x=overlay_df["time_s"],
                    y=overlay_df["hydrogen_pred_ci_upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            hydrogen_fig.add_trace(
                go.Scatter(
                    x=overlay_df["time_s"],
                    y=overlay_df["hydrogen_pred_ci_lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(51,209,122,0.18)",
                    name="H2 prediction band",
                    hoverinfo="skip",
                )
            )
        else:
            st.info(
                "Surrogate does not directly provide hydrogen mol% for this mapping. "
                "Displaying experimental hydrogen only."
            )
        hydrogen_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="Hydrogen mol%: Experiment vs Surrogate",
            xaxis_title="Time (s) used as tau proxy",
            yaxis_title="Hydrogen (mol%)",
        )
        chart_cols[1].plotly_chart(hydrogen_fig, width="stretch")

        st.dataframe(
            overlay_df[
                [
                    "time_s",
                    "temperature_c",
                    "methane_conversion_exp",
                    "methane_conversion_pred_mean",
                    "pred_ci_lower",
                    "pred_ci_upper",
                    "hydrogen_mol_percent_exp",
                    "hydrogen_mol_percent_pred_mean",
                    "is_out_of_distribution",
                    "ood_score",
                ]
            ],
            width="stretch",
        )

        report_cols = st.columns(2)
        validation_payload = {
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "dataset_path": dataset_path_label,
            "synora_version": SYNORA_VERSION,
            "metrics": {
                "rmse_conversion": float(validation_metrics["rmse_conversion"]),
                "mae_conversion": float(validation_metrics["mae_conversion"]),
                "bias_conversion": float(validation_metrics["bias_conversion"]),
            },
            "ood_stats": {
                "ood_count": int(validation_metrics["ood_count"]),
                "ood_fraction": float(validation_metrics["ood_fraction"]),
                "sample_count": int(validation_metrics["n_samples"]),
            },
            "validation_axis_note": VALIDATION_AXIS_NOTE,
        }
        with report_cols[0]:
            _json_download_button(
                "Download Validation Report",
                validation_payload,
                "validation_report",
            )
        with report_cols[1]:
            pdf_bytes = _pdf_summary_bytes(validation_payload)
            if pdf_bytes is not None:
                timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
                st.download_button(
                    "Download Validation PDF",
                    data=pdf_bytes,
                    file_name=f"validation_report_{timestamp}.pdf",
                    mime="application/pdf",
                )
            else:
                st.caption("Optional PDF summary unavailable. Install `fpdf2` to enable.")
