"""
Copyright 2023-2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
from pathlib import Path

import pytest
from dictdiffer import diff

from bluepyemodel.access_point.local import LocalAccessPoint
from tests.utils import DATA


def test_get_morphologies(db):
    morphology = db.get_morphologies()
    assert morphology["name"] == "C060114A5"
    assert Path(morphology["path"]).name == "C060114A5.asc"


def test_get_available_morphologies(db):
    names = db.get_available_morphologies()
    assert len(names) == 1
    assert list(names)[0] == "C060114A5"


def test_get_recipes(db):
    recipes = db.get_recipes()
    # json.dump(recipes, open(DATA / "test_recipes.json", "w"))
    expected_recipes = json.load(open(DATA / "test_recipes.json", "r"))
    assert list(diff(recipes, expected_recipes)) == []


def test_get_model_configuration(db):

    configuration = db.get_model_configuration()

    expected_parameters = json.load(open(DATA / "test_parameters.json", "r"))
    expected_mechanisms = json.load(open(DATA / "test_mechanisms.json", "r"))

    for p in configuration.parameters:
        assert p.location in expected_parameters["parameters"]
        for ep in expected_parameters["parameters"][p.location]:
            if ep["name"] == p.name and ep["val"] == p.value:
                break
        else:
            raise Exception("missing parameter")

    assert sorted(list(configuration.mechanism_names)) == [
        "CaDynamics_DC0",
        "Ca_HVA2",
        "Ca_LVAst",
        "Ih",
        "K_Pst",
        "K_Tst",
        "NaTg",
        "Nap_Et2",
        "SK_E2",
        "SKv3_1",
        "pas",
    ]


def test_get_final(db):
    final = db.get_final()
    assert "cADpyr_L5TPC" in final
    assert "parameters" in final["cADpyr_L5TPC"] or "params" in final["cADpyr_L5TPC"]


def test_load_pipeline_settings(db):
    assert db.pipeline_settings.path_extract_config == "tests/test_data/config/config_dict.json"
    assert db.pipeline_settings.validation_protocols == ["APWaveform_140"]


def test_get_model_name_for_final(db):
    db.emodel_metadata.iteration = ""
    assert db.get_model_name_for_final(seed=42) == "cADpyr_L5TPC__42"
    db.emodel_metadata.iteration = None
    assert db.get_model_name_for_final(seed=42) == "cADpyr_L5TPC__42"
    db.emodel_metadata.iteration = "hash"
    assert db.get_model_name_for_final(seed=42) == "cADpyr_L5TPC__hash__42"


def test_get_ion_currents_concentrations(db):
    expected_ion_currents = {
        "ica_Ca_HVA2",
        "ica_Ca_LVAst",
        "ik_K_Pst",
        "ik_K_Tst",
        "ina_NaTg",
        "ina_Nap_Et2",
        "ik_SK_E2",
        "ik_SKv3_1",
        "ihcn_Ih",
        "i_pas",
    }
    expected_ionic_concentrations = {
        "cai",
        "ki",
        "nai",
    }
    ion_currents, ionic_concentrations = db.get_ion_currents_concentrations()
    assert set(ion_currents) == expected_ion_currents
    assert set(ionic_concentrations) == expected_ionic_concentrations


def test_get_final_returns_empty_when_file_is_missing(db):
    db.final_path.unlink()

    assert db.get_final(lock_file=False) == {}


def test_get_final_recovers_from_temporary_file(db):
    db.final_path.write_text("{invalid json")
    temporary_path = db.final_path.with_name("final_tmp.json")
    expected = {"cADpyr_L5TPC": {"seed": 42}}
    temporary_path.write_text(json.dumps(expected))

    assert db.get_final(lock_file=False) == expected


def test_get_final_requires_a_final_path(db):
    db.final_path = None

    with pytest.raises(TypeError, match="Final_path is None"):
        db.get_final(lock_file=False)


def test_save_final_writes_primary_and_temporary_files(db, tmp_path):
    final_path = tmp_path / "results.json"
    expected = {"model": {"seed": 7}}

    db.save_final(expected, final_path, lock_file=False)

    assert json.loads(final_path.read_text()) == expected
    assert json.loads((tmp_path / "results_tmp.json").read_text()) == expected


def test_recipe_helpers_support_string_morphology(db, tmp_path, monkeypatch):
    morphology = tmp_path / "morphology.swc"
    morphology.write_text("morphology")
    monkeypatch.setattr(db, "emodel_dir", tmp_path)
    monkeypatch.setattr(
        db,
        "get_recipes",
        lambda: {"morph_path": ".", "morphology": str(morphology.name)},
    )

    assert db.morph_path == morphology
    assert db.get_morphologies() == {"name": "morphology", "path": str(morphology)}


def test_recipe_helpers_reject_unsupported_morphology(db, tmp_path, monkeypatch):
    morphology = tmp_path / "morphology.txt"
    morphology.write_text("morphology")
    monkeypatch.setattr(db, "emodel_dir", tmp_path)
    monkeypatch.setattr(
        db,
        "get_recipes",
        lambda: {"morph_path": ".", "morphology": str(morphology.name)},
    )

    with pytest.raises(FileNotFoundError, match="not supported"):
        _ = db.morph_path


def test_local_workflow_methods_are_noops(db):
    workflow = db.create_emodel_workflow(state="created")

    assert workflow.state == "created"
    assert db.get_emodel_workflow() == (None, None)
    assert db.check_emodel_workflow_configurations(workflow) is True
    assert db.store_or_update_emodel_workflow(workflow) is None


def test_add_entry_recipes_creates_and_updates_file(tmp_path):
    recipes_path = tmp_path / "nested" / "recipes.json"

    LocalAccessPoint.add_entry_recipes(
        recipes_path,
        "model_a",
        "morphology",
        ["cell", "cell.asc"],
        "params.json",
        "protocols.json",
        "features.json",
    )
    LocalAccessPoint.add_entry_recipes(
        recipes_path,
        "model_b",
        "morphology",
        ["cell", "cell.swc"],
        "params.json",
        "protocols.json",
        "features.json",
        pipeline_settings={"n_model": 2},
    )

    recipes = json.loads(recipes_path.read_text())
    assert recipes["model_a"]["morphology"] == [["cell", "cell.asc"]]
    assert recipes["model_a"]["pipeline_settings"]["n_model"] == 3
    assert recipes["model_b"]["pipeline_settings"] == {"n_model": 2}


def test_model_state_predicates(db, monkeypatch):
    db.emodel_metadata.iteration = "iteration"
    db.pipeline_settings.n_model = 2
    final = {
        "cADpyr_L5TPC__iteration__1": {
            "emodel": "cADpyr_L5TPC",
            "iteration": "iteration",
            "validated": True,
        },
        "cADpyr_L5TPC__iteration__2": {
            "emodel": "cADpyr_L5TPC",
            "iteration": "iteration",
            "validated": False,
        },
    }
    monkeypatch.setattr(db, "get_final_content", lambda: final)

    assert db.has_best_model(1) is True
    assert db.has_best_model(3) is False
    assert db.is_checked_by_validation(1) is True
    assert db.is_checked_by_validation(3) is False
    assert db.is_validated() is False

    final["cADpyr_L5TPC__iteration__2"]["validated"] = True
    assert db.is_validated() is True
