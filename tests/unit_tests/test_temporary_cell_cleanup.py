from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import bluepyemodel.evaluation.fitness_calculator_configuration as configuration_module


class FakeTemporaryCell:
    def __init__(self, *, instantiated=True):
        self.icell = object() if instantiated else None
        self.params = None
        self.mechanisms = None
        self.instantiate = Mock()
        self.destroy = Mock()


def test_destroy_temporary_cell_skips_uninstantiated_cell():
    cell = FakeTemporaryCell(instantiated=False)

    configuration_module.destroy_temporary_cell(cell, simulator=object())

    cell.destroy.assert_not_called()
    assert cell.params is None
    assert cell.mechanisms is None


def test_destroy_temporary_cell_clears_state_before_destroying():
    cell = FakeTemporaryCell()
    simulator = object()

    configuration_module.destroy_temporary_cell(cell, simulator)

    assert cell.params == {}
    assert cell.mechanisms == []
    cell.destroy.assert_called_once_with(sim=simulator)


def test_location_configuration_destroys_temporary_cell(monkeypatch):
    temporary_cell = FakeTemporaryCell()
    config = configuration_module.FitnessCalculatorConfiguration.__new__(
        configuration_module.FitnessCalculatorConfiguration
    )
    config._configure_locations_on_cell = Mock()
    monkeypatch.setattr(configuration_module, "deepcopy", lambda _: temporary_cell)

    config.configure_morphology_dependent_locations(
        _cell=SimpleNamespace(), simulator=object()
    )

    config._configure_locations_on_cell.assert_called_once()
    temporary_cell.destroy.assert_called_once()


def test_location_configuration_destroys_cell_when_configuration_fails(monkeypatch):
    temporary_cell = FakeTemporaryCell()
    config = configuration_module.FitnessCalculatorConfiguration.__new__(
        configuration_module.FitnessCalculatorConfiguration
    )
    config._configure_locations_on_cell = Mock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(configuration_module, "deepcopy", lambda _: temporary_cell)

    with pytest.raises(RuntimeError, match="boom"):
        config.configure_morphology_dependent_locations(
            _cell=SimpleNamespace(), simulator=object()
        )

    temporary_cell.destroy.assert_called_once()
