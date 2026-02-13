def test_import():
    import synora

    assert synora.__version__ == "0.1.0"


def test_layer1_public_api_imports() -> None:
    from synora.calibration.surrogate_fit import calibrated_predict, fit_surrogate
    from synora.economics.lcoh import EconInputs, hourly_economics
    from synora.generative.active_learning import run_active_learning
    from synora.generative.design_space import ReactorDesign
    from synora.generative.objectives import evaluate_design_surrogate
    from synora.generative.optimizer import propose_designs
    from synora.physics.label_pfr import PFRLabeler
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
    assert fit_surrogate is not None
    assert calibrated_predict is not None
    assert ReactorDesign is not None
    assert evaluate_design_surrogate is not None
    assert propose_designs is not None
    assert run_active_learning is not None
    assert PFRLabeler is not None
