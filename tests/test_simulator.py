from synora.twin.simulator import run_simulation


def test_run_simulation_returns_expected_columns() -> None:
    df = run_simulation(hours=24, methane_kg_per_hr=100.0, temp=800.0)

    expected_columns = {
        "time_hr",
        "methane_kg_per_hr",
        "temp_c",
        "conversion",
        "health",
        "h2_kg_per_hr",
        "carbon_kg_per_hr",
        "unreacted_methane_kg_per_hr",
        "cost_per_hr",
        "revenue_per_hr",
        "profit_per_hr",
        "lcoh_usd_per_kg",
        "cum_profit_usd",
    }

    assert len(df) == 24
    assert expected_columns.issubset(set(df.columns))
    assert (df["h2_kg_per_hr"] >= 0).all()
    assert (df["carbon_kg_per_hr"] >= 0).all()
    assert (df["profit_per_hr"].notna()).all()
