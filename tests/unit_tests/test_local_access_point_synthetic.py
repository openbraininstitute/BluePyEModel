"""Deterministic tests for LocalAccessPoint helpers using temporary directories."""

import json
from types import SimpleNamespace

import pytest

from bluepyemodel.access_point.local import LocalAccessPoint


def write_recipes(emodel_dir, emodel="L5PC", extra=None):
    recipes = {
        emodel: {
            "morph_path": "morphologies",
            "morphology": [["morph", "morph.asc"]],
            "params": "config/parameters.json",
            "features": "config/features.json",
            "protocol": "config/protocols.json",
            "pipeline_settings": {"n_model": 2},
        }
    }
    if extra:
        recipes[emodel].update(extra)
    recipes_path = emodel_dir / "recipes.json"
    recipes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(recipes_path, "w") as f:
        json.dump(recipes, f)
    return recipes_path


def make_access_point(
    tmp_path, monkeypatch, emodel="L5PC", extra=None, iteration_tag=None
):
    monkeypatch.chdir(tmp_path)
    emodel_dir = tmp_path / "emodel"
    recipes_path = write_recipes(emodel_dir, emodel=emodel, extra=extra)
    morph_dir = emodel_dir / "morphologies"
    morph_dir.mkdir(parents=True, exist_ok=True)
    (morph_dir / "morph.asc").write_text("morph")

    return LocalAccessPoint(
        emodel=emodel,
        emodel_dir=emodel_dir,
        recipes_path=recipes_path,
        iteration_tag=iteration_tag,
    )


def test_morph_dir_and_morph_path(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    assert access_point.morph_dir.name == "morphologies"
    assert access_point.morph_path.name == "morph.asc"


def test_morph_path_accepts_string_morphology(tmp_path, monkeypatch):
    access_point = make_access_point(
        tmp_path, monkeypatch, extra={"morphology": "morph.asc"}
    )

    assert access_point.morph_path.name == "morph.asc"


def test_morph_path_rejects_unsupported_extension(tmp_path, monkeypatch):
    access_point = make_access_point(
        tmp_path, monkeypatch, extra={"morphology": [["morph", "morph.txt"]]}
    )

    with pytest.raises(FileNotFoundError, match="not defined or not supported"):
        _ = access_point.morph_path


def test_morph_path_rejects_missing_file(tmp_path, monkeypatch):
    access_point = make_access_point(
        tmp_path, monkeypatch, extra={"morphology": [["other", "other.asc"]]}
    )

    with pytest.raises(FileNotFoundError, match="Morphology file not found"):
        _ = access_point.morph_path


def test_set_emodel_rejects_unknown_name(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="does not exist in the recipes"):
        access_point.set_emodel("absent")

    access_point.set_emodel("L5PC")
    assert access_point.emodel_metadata.emodel == "L5PC"


def test_load_pipeline_settings_adds_morph_modifiers(tmp_path, monkeypatch):
    access_point = make_access_point(
        tmp_path, monkeypatch, extra={"morph_modifiers": ["replace_axon_with_taper"]}
    )

    assert access_point.pipeline_settings.morph_modifiers == ["replace_axon_with_taper"]


def test_get_model_name_for_final_with_and_without_iteration(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)
    assert access_point.get_model_name_for_final(3) == "L5PC__3"

    iteration_dir = tmp_path / "iter"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    with_iteration = make_access_point(
        iteration_dir, monkeypatch, iteration_tag="iter1"
    )
    assert with_iteration.get_model_name_for_final(3) == "L5PC__iter1__3"


def test_get_final_returns_empty_when_missing(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    assert access_point.get_final() == {}


def test_get_final_raises_when_path_is_none(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)
    access_point.final_path = None

    with pytest.raises(TypeError, match="Final_path is None"):
        access_point.get_final()


def make_emodel(seed=1, scores=None):
    return SimpleNamespace(
        seed=seed,
        parameters={"gNa": 0.1},
        scores=scores
        if scores is not None
        else {"Step_150.soma.v.mean_frequency": 1.5},
        scores_validation={"IV_-40.soma.v.voltage_base": 0.5},
        features={"Step_150.soma.v.mean_frequency": 6.0},
        passed_validation=True,
        build_pdf_dependencies=lambda seed: ["figure.pdf"],
    )


def test_store_emodel_writes_final_entry(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    access_point.store_or_update_emodel(make_emodel(seed=2))

    final = access_point.get_final()
    entry = final["L5PC__2"]
    assert entry["seed"] == 2
    assert entry["validated"] is True
    assert entry["score"] == pytest.approx(1.5)
    assert entry["parameters"] == {"gNa": 0.1}
    assert entry["pdfs"] == ["figure.pdf"]


def test_store_emodel_overwrites_existing_entry(tmp_path, monkeypatch, caplog):
    access_point = make_access_point(tmp_path, monkeypatch)
    access_point.store_emodel(make_emodel(seed=2))

    access_point.store_emodel(make_emodel(seed=2, scores={"a": 9.0}))

    assert "already in the final.json" in caplog.text
    assert access_point.get_final()["L5PC__2"]["score"] == pytest.approx(9.0)


def test_has_best_model_and_validation_checks(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    assert access_point.has_best_model(seed=1) is False
    assert access_point.is_checked_by_validation(seed=1) is False
    assert access_point.is_validated() is False

    access_point.store_emodel(make_emodel(seed=1))
    assert access_point.has_best_model(seed=1) is True
    assert access_point.is_checked_by_validation(seed=1) is True
    assert access_point.is_validated() is False

    access_point.store_emodel(make_emodel(seed=2))
    assert access_point.is_validated() is True


def test_get_emodel_names_maps_entries(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)
    access_point.store_emodel(make_emodel(seed=1))

    assert access_point.get_emodel_names() == {"L5PC__1": "L5PC"}


def test_set_unfrozen_params_and_freeze(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)
    access_point.set_unfrozen_params(["gNa.somatic"])
    monkeypatch.setattr(
        access_point, "get_emodel", lambda: {"parameters": {"gK.somatic": 0.25}}
    )
    params = {
        "somatic": [
            {"name": "gNa", "val": [0.0, 1.0]},
            {"name": "gK", "val": [0.0, 1.0]},
            {"name": "gCa", "val": 0.3},
        ]
    }

    access_point._freeze_params(params)

    assert params["somatic"][0]["val"] == [0.0, 1.0]
    assert params["somatic"][1]["val"] == pytest.approx(0.25)
    assert params["somatic"][2]["val"] == pytest.approx(0.3)


def test_get_mechanisms_directory_and_available_mechanisms(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    assert access_point.get_mechanisms_directory() is None
    assert access_point.get_available_mechanisms() is None

    mech_dir = access_point.emodel_dir / "mechanisms"
    mech_dir.mkdir()
    (mech_dir / "Nav.mod").write_text(
        "NEURON {\n SUFFIX Nav\n USEION na READ ena WRITE ina\n}\n"
    )

    assert access_point.get_mechanisms_directory().name == "mechanisms"
    mechanisms = access_point.get_available_mechanisms()
    assert [m.name for m in mechanisms] == ["Nav"]


def test_get_available_morphologies(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    assert access_point.get_available_morphologies() == {"morph"}


def test_get_available_morphologies_returns_none_without_dir(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)
    access_point.__dict__["morph_dir"] = access_point.emodel_dir / "absent"

    assert access_point.get_available_morphologies() is None


def test_store_model_configuration_writes_json(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)
    configuration = SimpleNamespace(as_dict=lambda: {"parameters": []})
    target = access_point.emodel_dir / "out" / "configuration.json"

    access_point.store_model_configuration(configuration, path=target)

    with open(target) as f:
        assert json.load(f) == {"parameters": []}


def test_add_entry_recipes_creates_and_appends(tmp_path):
    recipes_path = tmp_path / "nested" / "recipes.json"

    LocalAccessPoint.add_entry_recipes(
        recipes_path,
        "L5PC",
        "morphologies",
        ["morph", "morph.asc"],
        "config/params.json",
        "config/protocols.json",
        "config/features.json",
    )
    LocalAccessPoint.add_entry_recipes(
        recipes_path,
        "L23PC",
        "morphologies",
        ["morph2", "morph2.asc"],
        "config/params2.json",
        "config/protocols2.json",
        "config/features2.json",
        pipeline_settings={"n_model": 5},
    )

    with open(recipes_path) as f:
        recipes = json.load(f)

    assert set(recipes) == {"L5PC", "L23PC"}
    assert recipes["L23PC"]["pipeline_settings"] == {"n_model": 5}
    assert isinstance(recipes["L5PC"]["pipeline_settings"], dict)


def test_unsupported_exports_raise(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    with pytest.raises(NotImplementedError):
        access_point.store_hocs()
    with pytest.raises(NotImplementedError):
        access_point.store_emodels_hoc()
    with pytest.raises(NotImplementedError):
        access_point.store_emodels_sonata()


def test_sonata_exists_reflects_exported_file(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)

    assert access_point.sonata_exists(seed=1) is False


def test_config_to_final_maps_fields(tmp_path, monkeypatch):
    access_point = make_access_point(tmp_path, monkeypatch)
    config = {
        "fitness": 2.5,
        "parameter": [{"name": "gNa", "value": 0.1}],
        "score": [{"name": "f", "value": 1.0}],
        "features": [{"name": "f", "value": 6.0}],
        "scoreValidation": [],
        "passedValidation": True,
        "seed": 4,
    }

    final = access_point._config_to_final(config)

    assert final["L5PC"]["score"] == pytest.approx(2.5)
    assert final["L5PC"]["seed"] == 4
    assert final["L5PC"]["validated"] is True


def test_get_final_content_prefers_recipes_final(tmp_path, monkeypatch, caplog):
    emodel_dir = tmp_path / "emodel"
    final_file = emodel_dir / "config" / "EM_final.json"
    final_file.parent.mkdir(parents=True, exist_ok=True)
    with open(final_file, "w") as f:
        json.dump(
            {
                "fitness": 1.0,
                "parameter": [],
                "score": [],
                "features": [],
                "scoreValidation": [],
                "passedValidation": False,
                "seed": 7,
            },
            f,
        )
    access_point = make_access_point(
        tmp_path, monkeypatch, extra={"final": "config/EM_final.json"}
    )
    access_point.final_path.write_text("{}")

    content = access_point.get_final_content()

    assert content["L5PC"]["seed"] == 7
    assert "using file from recipes" in caplog.text
