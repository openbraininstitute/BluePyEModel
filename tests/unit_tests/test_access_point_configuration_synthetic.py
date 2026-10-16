"""Synthetic coverage for LocalAccessPoint configuration persistence."""

import json
from types import SimpleNamespace

import bluepyemodel.access_point.local as local_module
from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata


def make_access_point(tmp_path, recipes):
    recipe_path = tmp_path / "recipes.json"
    recipe_path.write_text(json.dumps({"model": recipes}))
    access_point = object.__new__(LocalAccessPoint)
    access_point.emodel_dir = tmp_path
    access_point.recipes_path = recipe_path
    access_point.legacy_dir_structure = False
    access_point.with_seeds = False
    access_point.emodel_metadata = EModelMetadata(emodel="model", ttype="ttype")
    access_point.unfrozen_params = None
    return access_point


def test_load_pipeline_settings_variants(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, {})
    access_point.pipeline_settings = None
    recipes = {
        "pipeline_settings": {"validation_protocols": ["AP"]},
        "morph_modifiers": ["x"],
    }
    monkeypatch.setattr(access_point, "get_recipes", lambda: recipes)

    settings = access_point.load_pipeline_settings()

    assert settings.validation_protocols == ["AP"]
    assert settings.morph_modifiers == ["x"]

    recipes["pipeline_settings"] = "settings"
    monkeypatch.setattr(
        access_point,
        "get_json",
        lambda key: {"validation_protocols": ["Rin"], "morph_modifiers": []},
    )
    settings = access_point.load_pipeline_settings()
    assert settings.validation_protocols == ["Rin"]
    assert settings.morph_modifiers == []


def test_store_model_configuration_uses_explicit_and_recipe_paths(tmp_path):
    params_path = tmp_path / "nested" / "params.json"
    access_point = make_access_point(tmp_path, {"params": str(params_path)})
    configuration = SimpleNamespace(as_dict=lambda: {"parameters": {"g": 1}})

    explicit = tmp_path / "explicit" / "model.json"
    access_point.store_model_configuration(configuration, path=explicit)
    access_point.store_model_configuration(configuration)

    assert json.loads(explicit.read_text()) == {"parameters": {"g": 1}}
    assert json.loads((tmp_path / "nested/params.json").read_text()) == {
        "parameters": {"g": 1}
    }


def test_store_targets_configuration_and_load_optional_fields(tmp_path):
    target_path = tmp_path / "config" / "targets.json"
    access_point = make_access_point(tmp_path, {})
    access_point.pipeline_settings = SimpleNamespace(path_extract_config=target_path)
    configuration = SimpleNamespace(
        as_dict=lambda: {
            "files": [],
            "targets": [],
            "protocols_rheobase": {},
            "additional_fitness_efeatures": ["AP_amplitude"],
            "additional_fitness_protocols": ["APWaveform"],
            "protocols_mapping": {"APWaveform": "APWaveform"},
        }
    )

    access_point.store_targets_configuration(configuration)
    loaded = access_point.get_targets_configuration()

    assert json.loads(target_path.read_text())["emodel"] == "model"
    assert loaded.additional_fitness_efeatures == ["AP_amplitude"]
    assert loaded.additional_fitness_protocols == ["APWaveform"]
    assert loaded.protocols_mapping == {"APWaveform": "APWaveform"}


def test_store_fitness_configuration_and_load_modern_format(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, {"features": "config/features.json"})
    configuration = SimpleNamespace(as_dict=lambda: {"efeatures": [], "protocols": []})
    access_point.store_fitness_calculator_configuration(configuration)

    assert json.loads((tmp_path / "config/features.json").read_text()) == {
        "efeatures": [],
        "protocols": [],
    }

    captured = {}

    class FakeFitnessConfiguration:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        local_module, "FitnessCalculatorConfiguration", FakeFitnessConfiguration
    )
    access_point.pipeline_settings = SimpleNamespace(
        name_rmp_protocol="RMP",
        name_Rin_protocol="Rin",
        validation_protocols=[],
        stochasticity=False,
    )
    monkeypatch.setattr(
        access_point,
        "get_json",
        lambda key: {"efeatures": ["AP"], "protocols": ["APWaveform"]},
    )

    access_point.get_fitness_calculator_configuration()

    assert captured["efeatures"] == ["AP"]
    assert captured["protocols"] == ["APWaveform"]
    assert captured["name_rmp_protocol"] == "RMP"


def test_get_fitness_configuration_adds_ion_variables(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, {})
    captured = {}

    class FakeFitnessConfiguration:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        local_module, "FitnessCalculatorConfiguration", FakeFitnessConfiguration
    )
    access_point.pipeline_settings = SimpleNamespace(
        name_rmp_protocol=None,
        name_Rin_protocol=None,
        validation_protocols=[],
        stochasticity=False,
    )
    monkeypatch.setattr(
        access_point,
        "get_json",
        lambda key: {"efeatures": [], "protocols": []},
    )
    monkeypatch.setattr(
        access_point,
        "get_ion_currents_concentrations",
        lambda: (["ina"], ["cai"]),
    )

    access_point.get_fitness_calculator_configuration(record_ions_and_currents=True)

    assert captured["ion_variables"] == ["ina", "cai"]


def test_local_configuration_presence_predicates(tmp_path, monkeypatch):
    access_point = make_access_point(
        tmp_path,
        {
            "features": str(tmp_path / "features.json"),
            "params": str(tmp_path / "params.json"),
        },
    )
    access_point.pipeline_settings = SimpleNamespace(path_extract_config=None)
    assert access_point.has_pipeline_settings() is False
    assert access_point.has_fitness_calculator_configuration() is False
    assert access_point.has_model_configuration() is False
    assert access_point.has_targets_configuration() is None

    (tmp_path / "features.json").write_text("{}")
    (tmp_path / "params.json").write_text("{}")
    access_point.pipeline_settings = SimpleNamespace(
        path_extract_config=tmp_path / "targets.json"
    )
    (tmp_path / "targets.json").write_text("{}")
    monkeypatch.setattr(
        access_point,
        "get_recipes",
        lambda: {
            "pipeline_settings": {},
            "features": str(tmp_path / "features.json"),
            "params": str(tmp_path / "params.json"),
        },
    )

    assert access_point.has_pipeline_settings() is True
    assert access_point.has_fitness_calculator_configuration() is True
    assert access_point.has_model_configuration() is True
    assert access_point.has_targets_configuration() is True


def test_get_targets_configuration_defaults_optional_fields(tmp_path):
    access_point = make_access_point(tmp_path, {})
    target_path = tmp_path / "targets.json"
    target_path.write_text(
        json.dumps({"files": [], "targets": [], "protocols_rheobase": {}})
    )
    access_point.pipeline_settings = SimpleNamespace(path_extract_config=target_path)

    loaded = access_point.get_targets_configuration()

    assert loaded.additional_fitness_efeatures is None
    assert loaded.additional_fitness_protocols is None
    assert loaded.protocols_mapping is None
