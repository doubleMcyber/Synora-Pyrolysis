"""Synora package public API for the Layer 1 vertical slice."""

from synora.calibration.surrogate_fit import calibrate_and_store, calibrated_predict
from synora.economics.lcoh import EconInputs, hourly_economics
from synora.generative.active_learning import run_active_learning
from synora.generative.design_space import DesignBounds, ReactorDesign
from synora.generative.multizone import MultiZoneBounds, MultiZoneDesign, ZoneDesign
from synora.generative.objectives import evaluate_design_surrogate, evaluate_multizone_surrogate
from synora.generative.optimizer import propose_designs, propose_multizone_designs
from synora.generative.report import generate_design_report
from synora.reactor.model import (
    ReactorInputs,
    ReactorOutputs,
    ReactorState,
    apply_maintenance,
    simulate_step,
)
from synora.twin.simulator import build_simulation_context, build_visual_frame, run_simulation
from synora.validation.experimental import (
    DEFAULT_EXPERIMENTAL_DATASET_PATH,
    load_cv_reactor_experiment,
)
from synora.validation.metrics import compare_experiment_to_surrogate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ReactorInputs",
    "ReactorState",
    "ReactorOutputs",
    "simulate_step",
    "apply_maintenance",
    "EconInputs",
    "hourly_economics",
    "run_simulation",
    "build_simulation_context",
    "build_visual_frame",
    "calibrated_predict",
    "calibrate_and_store",
    "ReactorDesign",
    "DesignBounds",
    "MultiZoneDesign",
    "ZoneDesign",
    "MultiZoneBounds",
    "evaluate_design_surrogate",
    "evaluate_multizone_surrogate",
    "propose_designs",
    "propose_multizone_designs",
    "run_active_learning",
    "generate_design_report",
    "DEFAULT_EXPERIMENTAL_DATASET_PATH",
    "load_cv_reactor_experiment",
    "compare_experiment_to_surrogate",
]
