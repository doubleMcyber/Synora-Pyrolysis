from synora.twin.simulator import run_simulation


def test_run_simulation_returns_expected_columns() -> None:
    df = run_simulation(
        hours=48,
        methane_kg_per_hr=100.0,
        temp=980.0,
        residence_time_s=2.0,
        maintenance_interval_hr=24,
    )

    expected_columns = {
        "time_hr",
        "methane_kg_per_hr",
        "temp_c",
        "residence_time_s",
        "conversion",
        "health",
        "h2_kg_per_hr",
        "carbon_kg_per_hr",
        "carbon_rate_kg_per_hr",
        "fouling_rate_per_hr",
        "carbon_formation_index",
        "h2_yield_mol_per_mol_ch4",
        "unreacted_methane_kg_per_hr",
        "cost_per_hr",
        "revenue_per_hr",
        "profit_per_hr",
        "lcoh_usd_per_kg",
        "maintenance_event",
        "cum_profit_usd",
    }

    assert len(df) == 48
    assert expected_columns.issubset(set(df.columns))
    assert (df["h2_kg_per_hr"] >= 0).all()
    assert (df["carbon_kg_per_hr"] >= 0).all()
    assert (df["health"].between(0.0, 1.0)).all()
    assert df["maintenance_event"].sum() >= 1
    assert (df["profit_per_hr"].notna()).all()
