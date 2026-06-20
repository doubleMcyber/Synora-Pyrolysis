import pytest

from synora.economics.lcoh import EconInputs, hourly_economics


def test_hourly_economics_known_inputs() -> None:
    econ = EconInputs(
        methane_price_usd_per_kg=0.50,
        hydrogen_price_usd_per_kg=5.00,
        carbon_price_usd_per_kg=0.20,
        variable_opex_usd_per_hr=10.0,
        fixed_opex_usd_per_hr=5.0,
        capex_usd=0.0,
        capex_amortization_hours=1.0,
    )

    results = hourly_economics(
        h2_kg_per_hr=20.0,
        carbon_kg_per_hr=30.0,
        methane_kg_per_hr=100.0,
        econ_inputs=econ,
    )

    assert results["cost_per_hr"] == pytest.approx(65.0)
    assert results["revenue_per_hr"] == pytest.approx(106.0)
    assert results["profit_per_hr"] == pytest.approx(41.0)
    assert results["lcoh_usd_per_kg"] == pytest.approx(3.25)


def _econ(**overrides) -> EconInputs:
    base = dict(
        methane_price_usd_per_kg=0.50,
        hydrogen_price_usd_per_kg=5.00,
        carbon_price_usd_per_kg=0.20,
        variable_opex_usd_per_hr=10.0,
        fixed_opex_usd_per_hr=5.0,
        capex_usd=0.0,
        capex_amortization_hours=1.0,
    )
    base.update(overrides)
    return EconInputs(**base)


def test_capex_amortization_enters_cost() -> None:
    no_capex = hourly_economics(
        h2_kg_per_hr=20.0, carbon_kg_per_hr=30.0, methane_kg_per_hr=100.0, econ_inputs=_econ()
    )
    with_capex = hourly_economics(
        h2_kg_per_hr=20.0,
        carbon_kg_per_hr=30.0,
        methane_kg_per_hr=100.0,
        econ_inputs=_econ(capex_usd=8760.0, capex_amortization_hours=8760.0),  # 1.0 USD/hr
    )
    assert with_capex["cost_per_hr"] - no_capex["cost_per_hr"] == pytest.approx(1.0)


def test_lcoh_infinite_when_no_hydrogen() -> None:
    results = hourly_economics(
        h2_kg_per_hr=0.0, carbon_kg_per_hr=30.0, methane_kg_per_hr=100.0, econ_inputs=_econ()
    )
    assert results["lcoh_usd_per_kg"] == float("inf")
