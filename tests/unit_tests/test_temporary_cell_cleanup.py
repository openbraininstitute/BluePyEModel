from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import bluepyemodel.evaluation.fitness_calculator_configuration as configuration_module


class FakeTemporaryCell:
    def __init__(self, *, instantiated=True, instantiate_error=None):
        self.icell = None
        self.params = None
        self.mechanisms = None
        self.params_at_instantiate = None
        self.mechanisms_at_instantiate = None
        self.destroy = Mock()
        self._instantiated = instantiated
        self._instantiate_error = instantiate_error

    def instantiate(self, sim=None):
        self.params_at_instantiate = self.params
        self.mechanisms_at_instantiate = self.mechanisms
        if self._instantiate_error is not None:
            raise self._instantiate_error
        self.icell = object() if self._instantiated else None


def make_config(**kwargs):
    config = configuration_module.FitnessCalculatorConfiguration.__new__(
        configuration_module.FitnessCalculatorConfiguration
    )
    config._configure_locations_on_cell = Mock(**kwargs)
    return config


def test_location_configuration_destroys_temporary_cell(monkeypatch):
    temporary_cell = FakeTemporaryCell()
    config = make_config()
    monkeypatch.setattr(configuration_module, "deepcopy", lambda _: temporary_cell)

    simulator = object()
    config.configure_morphology_dependent_locations(_cell=SimpleNamespace(), simulator=simulator)

    assert temporary_cell.params_at_instantiate == {}
    assert temporary_cell.mechanisms_at_instantiate == []
    config._configure_locations_on_cell.assert_called_once()
    temporary_cell.destroy.assert_called_once_with(sim=simulator)


def test_location_configuration_destroys_cell_when_configuration_fails(monkeypatch):
    temporary_cell = FakeTemporaryCell()
    config = make_config(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(configuration_module, "deepcopy", lambda _: temporary_cell)

    with pytest.raises(RuntimeError, match="boom"):
        config.configure_morphology_dependent_locations(_cell=SimpleNamespace(), simulator=object())

    temporary_cell.destroy.assert_called_once()


def test_location_configuration_does_not_destroy_uninstantiated_cell(monkeypatch):
    temporary_cell = FakeTemporaryCell(instantiate_error=OSError("no morphology"))
    config = make_config()
    monkeypatch.setattr(configuration_module, "deepcopy", lambda _: temporary_cell)

    with pytest.raises(OSError, match="no morphology"):
        config.configure_morphology_dependent_locations(_cell=SimpleNamespace(), simulator=object())

    temporary_cell.destroy.assert_not_called()
    config._configure_locations_on_cell.assert_not_called()
