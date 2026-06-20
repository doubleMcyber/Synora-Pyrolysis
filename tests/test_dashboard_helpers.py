from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The 3D component module is import-safe (it only defines functions). The main dashboard
# app.py is NOT import-safe (it executes Streamlit at import), so its pure helpers
# (_hex_to_rgba, _trend_color, ...) are not unit-tested here.
_DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "apps" / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

reactor_3d = pytest.importorskip("reactor_3d_component")


def test_frame_payload_drops_nan_zone_temps_and_pads_to_three() -> None:
    payload = reactor_3d._frame_payload({"zone_temps_c": [900.0, float("nan"), 1000.0]})
    temps = payload["zone_temps_c"]
    assert len(temps) == 3
    assert all(t == t for t in temps)  # no NaN survived
    assert temps[0] == 900.0
    assert temps[1] == 1000.0  # NaN dropped; 1000 shifts up and the list pads to 3


def test_frame_payload_clamps_and_floors() -> None:
    payload = reactor_3d._frame_payload(
        {
            "conversion": 2.0,  # clamps to 1.0
            "confidence": -0.5,  # clamps to 0.0
            "h2_rate": -10.0,  # floored to 0.0
            "carbon_rate": -1.0,
            "is_out_of_distribution": 1.0,  # coerced to bool
        }
    )
    assert payload["conversion"] == 1.0
    assert payload["confidence"] == 0.0
    assert payload["h2_rate"] == 0.0
    assert payload["carbon_rate"] == 0.0
    assert isinstance(payload["is_out_of_distribution"], bool)
    assert payload["is_out_of_distribution"] is True


def test_frame_payload_frame_seq_format() -> None:
    payload = reactor_3d._frame_payload({"tick": 5, "time_hr": 2.5, "recenter_token": 1})
    assert payload["frame_seq"] == "5:2.50000:1"
