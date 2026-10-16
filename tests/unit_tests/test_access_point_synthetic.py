"""Synthetic coverage for local access-point filesystem branches."""

import json

import pytest

from bluepyemodel.access_point.access_point import DataAccessPoint, OptimisationState
from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata


def make_local_access_point(tmp_path, recipes):
    recipes_path = tmp_path / "recipes.json"
    recipes_path.write_text(json.dumps({"model": recipes}))

    access_point = object.__new__(LocalAccessPoint)
    access_point.emodel_dir = tmp_path
    access_point.recipes_path = recipes_path
    access_point.legacy_dir_structure = False
    access_point.with_seeds = False
    access_point.emodel_metadata = EModelMetadata(emodel="model")
    access_point.final_path = tmp_path / "final.json"
    access_point.new_final_path = None
    access_point.unfrozen_params = None
    return access_point


def test_local_recipe_paths_and_morphology(tmp_path):
    morphology_dir = tmp_path / "morphology"
    morphology_dir.mkdir()
    morphology = morphology_dir / "cell.ASC"
    morphology.write_text("morphology")
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"parameters": {}}))
    access_point = make_local_access_point(
        tmp_path,
        {
            "morphology": [["cell", morphology.name]],
            "params": params.name,
        },
    )

    assert access_point.get_json_path("params") == params
    assert access_point.morph_path == morphology
    assert access_point.get_morphologies() == {
        "name": "cell",
        "path": str(morphology),
    }
    assert access_point.get_available_morphologies() == {"cell"}


def test_local_recipe_seed_lookup_and_set_emodel(tmp_path):
    access_point = make_local_access_point(tmp_path, {})
    recipes_path = tmp_path / "recipes.json"
    recipes_path.write_text(json.dumps({"model": {}, "model_foo": {}}))
    access_point.recipes_path = recipes_path
    access_point.with_seeds = True
    access_point.emodel_metadata.emodel = "model_foo_42"

    access_point.set_emodel("model_foo_7")

    assert access_point.emodel_metadata.emodel == "model_foo"
    assert access_point.unfrozen_params is None
    with pytest.raises(ValueError, match="does not exist"):
        access_point.set_emodel("missing_7")


def test_local_final_read_write_and_recipe_conversion(tmp_path):
    access_point = make_local_access_point(tmp_path, {})
    expected = {"model": {"seed": 3}}

    assert access_point.get_final(lock_file=False) == {}
    access_point.save_final(expected, access_point.final_path, lock_file=False)
    assert access_point.get_final(lock_file=False) == expected

    access_point.final_path.write_text("invalid")
    temporary = tmp_path / "final_tmp.json"
    temporary.write_text(json.dumps({"recovered": True}))
    assert access_point.get_final(lock_file=False) == {"recovered": True}

    config = {
        "fitness": 1.5,
        "parameter": [{"name": "g", "val": 2}],
        "score": {"efeature": 0.2},
        "features": {"efeature": 0.1},
        "scoreValidation": {},
        "passedValidation": True,
        "seed": 3,
    }
    converted = access_point._config_to_final(config)
    assert converted["model"]["score"] == 1.5
    assert converted["model"]["parameters"] == config["parameter"]


def test_local_recipe_final_is_preferred(tmp_path, monkeypatch):
    access_point = make_local_access_point(tmp_path, {"final": "final_config.json"})
    config = {
        "fitness": 1.0,
        "parameter": [],
        "score": {},
        "features": {},
        "scoreValidation": {},
        "passedValidation": False,
        "seed": 1,
    }
    (tmp_path / "final_config.json").write_text(json.dumps(config))
    access_point.final_path.write_text(json.dumps({"ignored": True}))

    result = access_point.get_final_content(lock_file=False)

    assert result["model"]["seed"] == 1
    assert "ignored" not in result
    monkeypatch.setattr(access_point, "get_recipes", dict)
    assert access_point.get_final_content(lock_file=False) == {"ignored": True}


def test_local_emodel_formatting_and_filtering(tmp_path, monkeypatch):
    access_point = make_local_access_point(tmp_path, {})
    model_data = {
        "emodel": "model",
        "iteration_tag": "iter",
        "parameters": {"g": 1},
        "score": 2.0,
        "fitness": {"f": 0.1},
        "features": {},
        "validation_fitness": {},
        "validated": True,
        "seed": 1,
    }
    access_point.emodel_metadata.iteration = "iter"
    monkeypatch.setattr(
        access_point, "get_final_content", lambda **kwargs: {"model": model_data}
    )

    emodel = access_point.get_emodel(lock_file=False)

    assert emodel.emodel_metadata.iteration == "iter"
    assert emodel.parameters == {"g": 1}
    assert len(access_point.get_emodels()) == 1
    assert access_point.get_emodels()[0].parameters == emodel.parameters
    assert access_point.get_emodel_names() == {"model": "model"}
    assert access_point.get_emodel_etype_map() == {"model": "model"}


def test_local_configuration_and_mechanism_presence(tmp_path):
    access_point = make_local_access_point(
        tmp_path,
        {"features": "features.json", "params": "params.json"},
    )

    assert access_point.get_mechanisms_directory() is None
    assert access_point.get_available_mechanisms() is None
    assert access_point.has_fitness_calculator_configuration() is False
    assert access_point.has_model_configuration() is False

    (tmp_path / "features.json").write_text("{}")
    (tmp_path / "params.json").write_text("{}")
    assert access_point.has_fitness_calculator_configuration() is False
    assert access_point.has_model_configuration() is False


def test_data_access_point_ion_variables_and_optimisation_state(tmp_path, monkeypatch):
    access_point = DataAccessPoint("model")

    class Mechanism:
        def __init__(self):
            self.ionic_concentrations = ["cai"]

        def get_current(self):
            return ["ina", "i_pas"]

    monkeypatch.setattr(
        access_point, "get_available_mechanisms", lambda: [Mechanism(), Mechanism()]
    )
    currents, concentrations = access_point.get_ion_currents_concentrations()
    assert set(currents) == {"ina", "i_pas"}
    assert concentrations == ["cai"] or set(concentrations) == {"cai"}

    checkpoint = tmp_path / "checkpoint.pkl"
    monkeypatch.setattr(
        "bluepyemodel.access_point.access_point.get_checkpoint_path",
        lambda metadata, seed=None: checkpoint,
    )
    assert access_point.optimisation_state() == OptimisationState.EMPTY
    checkpoint.write_text("checkpoint")
    assert access_point.optimisation_state() == OptimisationState.COMPLETED


def test_local_export_methods_are_explicitly_unsupported(tmp_path):
    access_point = make_local_access_point(tmp_path, {})

    with pytest.raises(NotImplementedError):
        access_point.store_emodels_hoc()
    with pytest.raises(NotImplementedError):
        access_point.store_emodels_sonata()
