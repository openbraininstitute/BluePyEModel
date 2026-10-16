"""Synthetic tests for local evaluation response persistence."""

from types import SimpleNamespace

import numpy

from bluepyemodel.evaluation.evaluation import (
    check_local_responses_presence,
    fill_initial_parameters,
    load_responses_from_local_files,
    locally_store_responses,
)


def _emodel(seed=1):
    return SimpleNamespace(
        seed=seed,
        emodel_metadata=SimpleNamespace(as_string=lambda value: f"model__{value}"),
    )


def test_locally_store_responses_writes_traces_and_scalars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    emodel = _emodel()
    emodel.responses = {
        "Step.soma.v": {"time": [0.0, 1.0], "voltage": [-80.0, -70.0]},
        "bpo_threshold_current": 0.2,
        "Step.soma.i": {"time": None, "voltage": None},
    }

    locally_store_responses(emodel)

    output_dir = tmp_path / "recordings" / "model__1"
    trace = numpy.loadtxt(output_dir / "Step.soma.v.dat")
    scalar = numpy.loadtxt(output_dir / "bpo_threshold_current.dat")
    assert trace.tolist() == [[0.0, -80.0], [1.0, -70.0]]
    assert scalar.tolist() == 0.2
    assert not (output_dir / "Step.soma.i.dat").exists()


def test_check_local_responses_presence_requires_only_voltage_files(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    emodel = _emodel()
    recording = SimpleNamespace(name="Step.soma.v", variable="v")
    current = SimpleNamespace(name="Step.soma.ina", variable="ina")
    cell_eval = SimpleNamespace(
        fitness_protocols={
            "main_protocol": SimpleNamespace(
                protocols={"Step": SimpleNamespace(recordings=[recording, current])}
            )
        }
    )

    assert check_local_responses_presence([emodel], cell_eval) is False
    output_dir = tmp_path / "recordings" / "model__1"
    output_dir.mkdir(parents=True)
    (output_dir / "Step.soma.v.dat").write_text("0 -80\n1 -70\n")

    assert check_local_responses_presence([emodel], cell_eval) is True


def test_load_responses_from_local_files_loads_traces_scalars_and_copies_evaluator(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    emodel = _emodel()
    output_dir = tmp_path / "recordings" / "model__1"
    output_dir.mkdir(parents=True)
    numpy.savetxt(output_dir / "Step.soma.v.dat", [[0.0, -80.0], [1.0, -70.0]])
    numpy.savetxt(output_dir / "bpo_holding_current.dat", [0.1])
    evaluator = SimpleNamespace(label="original")

    responses = load_responses_from_local_files([emodel], evaluator)[0]

    assert responses["Step.soma.v"].name == "Step"
    numpy.testing.assert_allclose(responses["Step.soma.v"].response["time"], [0.0, 1.0])
    numpy.testing.assert_allclose(
        responses["Step.soma.v"].response["voltage"], [-80.0, -70.0]
    )
    numpy.testing.assert_allclose(responses["bpo_holding_current"], [0.1])
    assert responses["evaluator"] is not evaluator
    assert responses["evaluator"].label == "original"


def test_fill_initial_parameters_replaces_only_unbounded_unset_parameters():
    replaceable = SimpleNamespace(name="gNa", bounds=None, _value=None, frozen=False)
    bounded = SimpleNamespace(name="gK", bounds=(0.0, 1.0), _value=None, frozen=False)
    valued = SimpleNamespace(name="gCa", bounds=None, _value=0.3, frozen=False)
    evaluator = SimpleNamespace(
        cell_model=SimpleNamespace(
            params={"gNa": replaceable, "gK": bounded, "gCa": valued}
        ),
        params=[replaceable, bounded, valued],
        param_names=["gNa", "gK", "gCa"],
    )

    fill_initial_parameters(evaluator, {"gNa": 0.7, "gK": 0.2, "gCa": 0.4})

    assert replaceable._value == 0.7
    assert replaceable.frozen is True
    assert evaluator.params == [bounded, valued]
    assert evaluator.param_names == ["gK", "gCa"]
    assert bounded._value is None
    assert valued._value == 0.3
