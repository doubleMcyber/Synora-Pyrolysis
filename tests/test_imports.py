def test_import():
    import synora

    assert synora.__version__ == "0.1.0"


def test_layer1_public_api_imports() -> None:
    from synora.economics.lcoh import EconInputs, hourly_economics
    from synora.reactor.model import (
        ReactorInputs,
        ReactorOutputs,
        ReactorState,
        apply_maintenance,
        simulate_step,
    )
    from synora.twin.simulator import run_simulation

    assert ReactorInputs is not None
    assert ReactorState is not None
    assert ReactorOutputs is not None
    assert simulate_step is not None
    assert apply_maintenance is not None
    assert EconInputs is not None
    assert hourly_economics is not None
    assert run_simulation is not None
