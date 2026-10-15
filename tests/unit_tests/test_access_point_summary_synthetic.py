"""Deterministic tests for DataAccessPoint summary, state and mechanism helpers."""

from types import SimpleNamespace

import pytest

from bluepyemodel.access_point import access_point as ap_module
from bluepyemodel.access_point.access_point import DataAccessPoint, OptimisationState


class FakeAccessPoint(DataAccessPoint):
    """Concrete access point exercising only the deterministic base-class behaviour."""

    def __init__(self, emodels=None, mechanisms=None, settings=None, metadata=None):
        self.emodel_metadata = metadata or SimpleNamespace(
            emodel="L5PC",
            iteration=None,
            as_string=lambda seed=None: "L5PC__meta",
        )
        self._emodels = emodels if emodels is not None else []
        self._mechanisms = mechanisms
        self.pipeline_settings = settings or SimpleNamespace(
            optimiser="IBEA", max_ngen=10
        )

    def get_emodels(self):  # pylint: disable=arguments-differ
        return self._emodels

    def get_available_mechanisms(self):
        return self._mechanisms

    def has_pipeline_settings(self):
        return True

    def has_targets_configuration(self):
        return False

    def has_fitness_calculator_configuration(self):
        return True

    def has_model_configuration(self):
        return True


def test_set_emodel_and_default_pipeline_settings():
    access_point = FakeAccessPoint()

    access_point.set_emodel("new_emodel")

    assert access_point.emodel_metadata.emodel == "new_emodel"
    assert access_point.get_pipeline_settings() is not None


def test_get_available_efeatures_cleaned_and_raw():
    access_point = FakeAccessPoint()

    raw = access_point.get_available_efeatures(cleaned=False)
    cleaned = access_point.get_available_efeatures()

    assert "AP_begin_time" in raw
    assert "AP_begin_time" not in cleaned
    assert "peak_time" not in cleaned
    assert not any(f.endswith("indices") for f in cleaned)
    assert "voltage" not in cleaned


def test_get_available_traces_and_unimplemented_hooks():
    access_point = FakeAccessPoint()

    assert access_point.get_available_traces() is None
    assert access_point.get_distributions() is None
    assert access_point.get_emodel() is None
    assert access_point.store_efeatures({}, {}) is None
    assert access_point.store_protocols({}) is None
    assert access_point.store_targets_configuration() is None
    assert access_point.get_targets_configuration() is None
    assert access_point.store_model_configuration() is None
    assert access_point.update_emodel_images(seed=1) is None


def test_get_ion_currents_concentrations_without_mechanisms():
    access_point = FakeAccessPoint(mechanisms=None)

    assert access_point.get_ion_currents_concentrations() == (None, None)


def test_get_ion_currents_concentrations_collects_unique_values():
    mechanisms = [
        SimpleNamespace(get_current=lambda: ["ina"], ionic_concentrations=["nai"]),
        SimpleNamespace(get_current=lambda: ["ina", "ik"], ionic_concentrations=["ki"]),
    ]
    access_point = FakeAccessPoint(mechanisms=mechanisms)

    currents, concentrations = access_point.get_ion_currents_concentrations()

    assert sorted(currents) == ["i_pas", "ik", "ina"]
    assert sorted(concentrations) == ["ki", "nai"]


def test_optimisation_state_empty_when_no_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ap_module,
        "get_checkpoint_path",
        lambda metadata, seed=None: str(tmp_path / "absent.pkl"),
    )

    assert FakeAccessPoint().optimisation_state() == OptimisationState.EMPTY


def test_optimisation_state_completed_without_continue(monkeypatch, tmp_path):
    checkpoint = tmp_path / "chk.pkl"
    checkpoint.write_text("x")
    monkeypatch.setattr(
        ap_module, "get_checkpoint_path", lambda metadata, seed=None: str(checkpoint)
    )

    assert FakeAccessPoint().optimisation_state() == OptimisationState.COMPLETED


def test_optimisation_state_cma_in_progress_and_completed(monkeypatch, tmp_path):
    checkpoint = tmp_path / "chk.pkl"
    checkpoint.write_text("x")
    monkeypatch.setattr(
        ap_module, "get_checkpoint_path", lambda metadata, seed=None: str(checkpoint)
    )
    settings = SimpleNamespace(optimiser="SO-CMA", max_ngen=10)

    active = SimpleNamespace(active=True, check_termination=lambda gen: None)
    monkeypatch.setattr(
        ap_module,
        "read_checkpoint",
        lambda path: ({"generation": 3, "CMA_es": active}, 1),
    )
    assert (
        FakeAccessPoint(settings=settings).optimisation_state(continue_opt=True)
        == OptimisationState.IN_PROGRESS
    )

    finished = SimpleNamespace(active=False, check_termination=lambda gen: None)
    monkeypatch.setattr(
        ap_module,
        "read_checkpoint",
        lambda path: ({"generation": 9, "CMA_es": finished}, 1),
    )
    assert (
        FakeAccessPoint(settings=settings).optimisation_state(continue_opt=True)
        == OptimisationState.COMPLETED
    )


def test_optimisation_state_ibea_progress_and_completion(monkeypatch, tmp_path):
    checkpoint = tmp_path / "chk.pkl"
    checkpoint.write_text("x")
    monkeypatch.setattr(
        ap_module, "get_checkpoint_path", lambda metadata, seed=None: str(checkpoint)
    )
    settings = SimpleNamespace(optimiser="IBEA", max_ngen=10)

    monkeypatch.setattr(
        ap_module, "read_checkpoint", lambda path: ({"generation": 2}, 1)
    )
    assert (
        FakeAccessPoint(settings=settings).optimisation_state(continue_opt=True)
        == OptimisationState.IN_PROGRESS
    )

    monkeypatch.setattr(
        ap_module, "read_checkpoint", lambda path: ({"generation": 20}, 1)
    )
    assert (
        FakeAccessPoint(settings=settings).optimisation_state(continue_opt=True)
        == OptimisationState.COMPLETED
    )


def test_optimisation_state_rejects_unknown_optimiser(monkeypatch, tmp_path):
    checkpoint = tmp_path / "chk.pkl"
    checkpoint.write_text("x")
    monkeypatch.setattr(
        ap_module, "get_checkpoint_path", lambda metadata, seed=None: str(checkpoint)
    )
    monkeypatch.setattr(
        ap_module, "read_checkpoint", lambda path: ({"generation": 1}, 1)
    )
    settings = SimpleNamespace(optimiser="UNKNOWN", max_ngen=10)

    with pytest.raises(ValueError, match="Unknown optimiser"):
        FakeAccessPoint(settings=settings).optimisation_state(continue_opt=True)


def test_str_reports_metadata_configuration_and_no_emodels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    access_point = FakeAccessPoint()

    text = str(access_point)

    assert "SUMMARY: EMODEL CREATION" in text
    assert "emodel: L5PC" in text
    assert "Has pipeline settings: True" in text
    assert "Has targets configuration: False" in text
    assert "No emodels" in text
    assert "OPTIMISATION STATUS" not in text


def test_str_reports_checkpoints_and_best_emodel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "L5PC__meta__seed=1.pkl"
    checkpoint.write_bytes(b"placeholder")
    logbook = SimpleNamespace(select=lambda field: [0, 1, 2])
    hall_of_fame = [SimpleNamespace(fitness=SimpleNamespace(values=[1.0, 2.0]))]
    monkeypatch.setattr(
        ap_module,
        "read_checkpoint",
        lambda path: ({"logbook": logbook, "halloffame": hall_of_fame}, 1),
    )

    emodels = [
        SimpleNamespace(seed=1, fitness=5.0, passed_validation=True),
        SimpleNamespace(seed=2, fitness=2.0, passed_validation=False),
    ]
    access_point = FakeAccessPoint(emodels=emodels)

    text = str(access_point)

    assert "Number of checkpoints: 1" in text
    assert "Seed 1;" in text
    assert "Last generation: 2;" in text
    assert "Best fitness: 3.0" in text
    assert "Emodels stored: 2" in text
    assert "Number of validated emodel: 1" in text
    assert "Best emodel: seed 2" in text
