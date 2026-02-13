from synora.reactor.model import ReactorInputs, ReactorState, apply_maintenance, simulate_step


def test_simulate_step_returns_non_negative_outputs() -> None:
    inputs = ReactorInputs(
        methane_kg_per_hr=100.0,
        temp_c=980.0,
        residence_time_s=2.0,
        max_conversion=0.90,
    )
    state = ReactorState(conversion=0.0, health=1.0, hours_operated=0.0)

    next_state, outputs = simulate_step(inputs, state)

    assert outputs.h2_kg_per_hr >= 0
    assert outputs.carbon_kg_per_hr >= 0
    assert outputs.carbon_rate_kg_per_hr >= 0
    assert outputs.fouling_rate_per_hr >= 0
    assert outputs.unreacted_methane_kg_per_hr >= 0
    assert 0 <= outputs.conversion <= inputs.max_conversion
    assert 0 <= next_state.health <= 1


def test_health_decreases_with_fouling() -> None:
    inputs = ReactorInputs(
        methane_kg_per_hr=120.0,
        temp_c=1080.0,
        residence_time_s=4.5,
        max_conversion=0.95,
    )
    initial = ReactorState(conversion=0.0, health=1.0, hours_operated=0.0)

    state_1, outputs_1 = simulate_step(inputs, initial)
    state_2, outputs_2 = simulate_step(inputs, state_1)

    assert outputs_1.fouling_rate_per_hr > 0
    assert state_1.health < initial.health
    assert state_2.health <= state_1.health
    assert outputs_2.conversion < outputs_1.conversion


def test_maintenance_restores_health() -> None:
    degraded_state = ReactorState(conversion=0.41, health=0.52, hours_operated=120.0)
    restored = apply_maintenance(degraded_state, restored_health=0.96)

    assert restored.health == 0.96
    assert restored.hours_operated == degraded_state.hours_operated
    assert restored.conversion > degraded_state.conversion
