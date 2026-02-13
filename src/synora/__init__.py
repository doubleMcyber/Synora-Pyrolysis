"""Synora package public API for the Layer 1 vertical slice."""

from synora.calibration.surrogate_fit import calibrate_and_store, calibrated_predict
from synora.economics.lcoh import EconInputs, hourly_economics
from synora.reactor.model import (
    ReactorInputs,
    ReactorOutputs,
    ReactorState,
    apply_maintenance,
    simulate_step,
)
from synora.twin.simulator import run_simulation

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
    "calibrated_predict",
    "calibrate_and_store",
]
