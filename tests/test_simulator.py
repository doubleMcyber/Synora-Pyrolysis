from synora.twin.simulator import build_visual_frame, run_simulation


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


def test_build_visual_frame_maps_required_keys_and_ranges() -> None:
    df = run_simulation(
        hours=2,
        methane_kg_per_hr=95.0,
        temp=970.0,
        residence_time_s=1.7,
        ticks_per_hour=4,
        zone_count=3,
    )
    row = df.iloc[-1]
    frame = build_visual_frame(row, zone_count=3)

    required = {
        "zone_temps_c",
        "conversion",
        "h2_rate",
        "carbon_rate",
        "fouling_index",
        "deltaP_kpa",
        "power_kw",
        "is_out_of_distribution",
        "ood_score",
        "confidence",
        "methane_fraction",
        "hydrogen_fraction",
    }
    assert required.issubset(set(frame.keys()))
    assert isinstance(frame["zone_temps_c"], list)
    assert len(frame["zone_temps_c"]) >= 1
    assert all(value > 0 for value in frame["zone_temps_c"])
    assert frame["h2_rate"] >= 0
    assert frame["carbon_rate"] >= 0
    assert frame["deltaP_kpa"] >= 0
    assert frame["power_kw"] >= 0
    assert 0.0 <= frame["conversion"] <= 1.0
    assert frame["fouling_index"] >= 0
    assert 0.0 <= frame["confidence"] <= 1.0
