"""Deterministic tests for export_emodel and validation modules."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import pytest

from bluepyemodel.export_emodel import export_emodel as export_module
from bluepyemodel.validation import validation as validation_module
from bluepyemodel.validation import validation_functions


def make_metadata(**overrides):
    values = {
        "emodel": "L5PC",
        "brain_region": "SSCX",
        "etype": "cAD",
        "mtype": "L5_TPC",
        "synapse_class": "EXC",
        "iteration": None,
    }
    values.update(overrides)
    return SimpleNamespace(
        as_string=lambda seed=None, use_allen_notation=True: "meta", **values
    )


def make_emodel(responses=None, **overrides):
    values = {
        "emodel_metadata": make_metadata(),
        "responses": responses if responses is not None else {},
        "parameters": {"gnabar": 0.1},
        "passed_validation": True,
        "seed": 3,
        "fitness": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_write_node_file_writes_full_population(tmp_path):
    emodel = make_emodel(
        responses={"bpo_holding_current": -0.05, "bpo_threshold_current": 0.2},
    )
    node_file = tmp_path / "nodes.h5"
    morphology = tmp_path / "morph.asc"
    morphology.write_text("morph")

    export_module._write_node_file(
        emodel, tmp_path / "model.hoc", node_file, morphology
    )

    with h5py.File(node_file, "r") as f:
        population = f["/nodes/SSCX_neurons/0"]
        assert population["model_type"][()].decode() == "biophysical"
        assert population["morphology"][()].decode() == "morph"
        assert population["etype"][()].decode() == "cAD"
        assert population["region"][()].decode() == "SSCX"
        assert population["dynamics_params/holding_current"][()] == pytest.approx(-0.05)
        assert population["dynamics_params/threshold_current"][()] == pytest.approx(0.2)


def test_write_node_file_omits_optional_fields(tmp_path):
    emodel = make_emodel(
        emodel_metadata=make_metadata(
            brain_region=None, etype=None, mtype=None, synapse_class=None
        ),
        responses={"bpo_holding_current": 1.0},
    )
    node_file = tmp_path / "nodes.h5"

    export_module._write_node_file(emodel, "model.hoc", node_file)

    with h5py.File(node_file, "r") as f:
        population = f["/nodes/None_neurons/0"]
        assert "morphology" not in population
        assert "etype" not in population
        assert "region" not in population
        assert "dynamics_params" not in population


def test_write_hoc_file_renames_model_and_writes_content(tmp_path):
    cell_model = SimpleNamespace(
        name="old",
        create_hoc=lambda param_values, template, template_dir: "hoc-content",
    )
    emodel = make_emodel()
    hoc_path = tmp_path / "model.hoc"

    export_module._write_hoc_file(
        cell_model, emodel, hoc_path, new_emodel_name="new_name"
    )

    assert cell_model.name == "new_name"
    assert hoc_path.read_text() == "hoc-content"


def make_cell_model(morphology_path):
    return SimpleNamespace(
        name="cell",
        morphology=SimpleNamespace(
            morphology_path=str(morphology_path), morph_modifiers=[]
        ),
        create_hoc=lambda param_values, template, template_dir: f"hoc:{template}",
    )


def test_export_model_sonata_creates_artifacts(tmp_path):
    morphology = tmp_path / "morph.asc"
    morphology.write_text("morph")
    cell_model = make_cell_model(morphology)
    emodel = make_emodel(
        responses={"bpo_holding_current": 0.0, "bpo_threshold_current": 0.1},
        passed_validation=False,
    )
    output_dir = tmp_path / "out"

    export_module._export_model_sonata(
        cell_model, emodel, output_dir=output_dir, new_emodel_name="renamed"
    )

    assert emodel.emodel_metadata.emodel == "renamed"
    assert (output_dir / "morph.asc").is_file()
    assert (output_dir / "nodes.h5").is_file()
    assert (
        "cell_template_neurodamus_sbo.jinja2" in (output_dir / "model.hoc").read_text()
    )


def test_export_emodel_hoc_creates_hoc_and_morphology(tmp_path):
    morphology = tmp_path / "morph.swc"
    morphology.write_text("morph")
    cell_model = make_cell_model(morphology)
    emodel = make_emodel(passed_validation=False)
    output_dir = tmp_path / "hoc_out"

    export_module._export_emodel_hoc(cell_model, emodel, output_dir=output_dir)

    assert (output_dir / "morph.swc").is_file()
    assert "cell_template.jinja2" in (output_dir / "model.hoc").read_text()


def test_export_emodels_hoc_applies_metadata_and_modifiers(tmp_path, monkeypatch):
    morphology = tmp_path / "morph.asc"
    morphology.write_text("morph")
    cell_model = make_cell_model(morphology)
    cell_evaluator = SimpleNamespace(cell_model=cell_model)
    emodel = make_emodel()
    new_metadata = make_metadata(emodel="other")
    access_point = MagicMock()
    access_point.emodel_metadata = make_metadata()
    access_point.get_emodels.return_value = [emodel]

    monkeypatch.setattr(
        export_module, "get_evaluator_from_access_point", lambda *a, **k: cell_evaluator
    )
    monkeypatch.setattr(export_module, "select_emodels", lambda *a, **k: [emodel])
    calls = []
    monkeypatch.setattr(
        export_module,
        "_export_emodel_hoc",
        lambda model, mo, output_dir=None, new_emodel_name=None: calls.append(mo),
    )

    export_module.export_emodels_hoc(access_point, new_metadata=new_metadata)

    assert calls == [emodel]
    assert emodel.emodel_metadata is new_metadata
    assert cell_model.morphology.morph_modifiers is None


def test_export_emodels_hoc_stops_without_selected_emodels(monkeypatch):
    access_point = MagicMock()
    access_point.emodel_metadata = make_metadata()
    access_point.get_emodels.return_value = []
    monkeypatch.setattr(
        export_module,
        "get_evaluator_from_access_point",
        lambda *a, **k: SimpleNamespace(cell_model=MagicMock()),
    )
    monkeypatch.setattr(export_module, "select_emodels", lambda *a, **k: [])
    export_hoc = MagicMock()
    monkeypatch.setattr(export_module, "_export_emodel_hoc", export_hoc)

    export_module.export_emodels_hoc(access_point)

    export_hoc.assert_not_called()


def make_access_point(**settings):
    defaults = {
        "validation_function": None,
        "validation_protocols": ["Step_150"],
        "validation_threshold": 5.0,
    }
    defaults.update(settings)
    return SimpleNamespace(
        pipeline_settings=SimpleNamespace(**defaults),
        emodel_metadata=make_metadata(),
        store_or_update_emodel=lambda model: None,
    )


def test_define_validation_function_variants():
    assert (
        validation_module.define_validation_function(make_access_point())
        is validation_functions.validate_max_score
    )
    assert (
        validation_module.define_validation_function(
            make_access_point(validation_function="max_score")
        )
        is validation_functions.validate_max_score
    )
    assert (
        validation_module.define_validation_function(
            make_access_point(validation_function="mean_score")
        )
        is validation_functions.validate_mean_score
    )
    with pytest.raises(ValueError, match="must be 'max_score' or 'mean_score'"):
        validation_module.define_validation_function(
            make_access_point(validation_function="unknown")
        )
    with pytest.raises(TypeError, match="not callable"):
        validation_module.define_validation_function(
            make_access_point(validation_function=123)
        )


def test_define_validation_function_loads_from_file(tmp_path):
    module_file = tmp_path / "custom_validation.py"
    module_file.write_text(
        "def custom(model, threshold, only_validation):\n    return True\n"
    )

    function = validation_module.define_validation_function(
        make_access_point(validation_function=[str(module_file), "custom"])
    )

    assert function(None, 1.0, False) is True


def test_define_validation_function_accepts_callable():
    def callable_validation(model, threshold, only_validation):
        return False

    assert (
        validation_module.define_validation_function(
            make_access_point(validation_function=callable_validation)
        )
        is callable_validation
    )


def test_compute_scores_splits_validation_and_optimisation():
    fitness_calculator = SimpleNamespace(
        calculate_values=lambda responses: {
            "Step_150.soma.v.mean_frequency": [1.0, None, 3.0],
            "Step_200.soma.v.mean_frequency": None,
        },
        calculate_scores=lambda responses: {
            "Step_150.soma.v.mean_frequency": 0.5,
            "Step_200.soma.v.mean_frequency": 1.5,
        },
    )
    model = SimpleNamespace(
        evaluator=SimpleNamespace(fitness_calculator=fitness_calculator),
        responses={},
        features={},
        scores={},
        scores_validation={},
    )

    validation_module.compute_scores(model, ["Step_150"])

    assert model.features["Step_150.soma.v.mean_frequency"] == pytest.approx(2.0)
    assert model.features["Step_200.soma.v.mean_frequency"] is None
    assert model.scores_validation == {"Step_150.soma.v.mean_frequency": 0.5}
    assert model.scores == {"Step_200.soma.v.mean_frequency": 1.5}


def test_validate_returns_empty_without_emodels(monkeypatch):
    monkeypatch.setattr(
        validation_module,
        "get_evaluator_from_access_point",
        lambda *a, **k: MagicMock(),
    )
    monkeypatch.setattr(validation_module, "compute_responses", lambda *a, **k: [])

    assert validation_module.validate(make_access_point(), map) == []


def test_validate_stores_validated_models(monkeypatch):
    stored = []
    access_point = make_access_point()
    access_point.store_or_update_emodel = stored.append
    model = SimpleNamespace(
        responses={"bpo_holding_current": 0.1, "Step_150.soma.v": [1.0]},
        scores={},
        scores_validation={},
        features={},
        passed_validation=None,
        threshold_data=None,
    )

    monkeypatch.setattr(
        validation_module,
        "get_evaluator_from_access_point",
        lambda *a, **k: MagicMock(),
    )
    monkeypatch.setattr(validation_module, "compute_responses", lambda *a, **k: [model])
    monkeypatch.setattr(validation_module, "compute_scores", lambda m, protocols: None)
    monkeypatch.setattr(
        validation_module, "define_validation_function", lambda ap: lambda *args: True
    )

    result = validation_module.validate(
        access_point, map, preselect_for_validation=True
    )

    assert result == [model]
    assert model.passed_validation is True
    assert model.threshold_data == {"bpo_holding_current": 0.1}
    assert stored == [model]
