from synora.reactor.model import ReactorInputs, ReactorState, simulate_step


def test_simulate_step_returns_non_negative_outputs() -> None:
    inputs = ReactorInputs(methane_kg_per_hr=100.0, temp_c=800.0, max_conversion=0.90)
    state = ReactorState(conversion=0.0, health=1.0, hours_operated=0.0)

    next_state, outputs = simulate_step(inputs, state)

    assert outputs.h2_kg_per_hr >= 0
    assert outputs.carbon_kg_per_hr >= 0
    assert outputs.unreacted_methane_kg_per_hr >= 0
    assert 0 <= outputs.conversion <= inputs.max_conversion
    assert 0 <= next_state.health <= 1
