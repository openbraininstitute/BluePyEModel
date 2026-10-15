"""Synthetic coverage for target-configuration orchestration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.efeatures_extraction.targets_configurator import TargetsConfigurator


def test_new_configuration_replaces_existing_configuration(monkeypatch):
    access_point = MagicMock()
    access_point.get_available_traces.return_value = ["trace"]
    access_point.get_available_efeatures.return_value = ["feature"]
    configurator = TargetsConfigurator(access_point, configuration=MagicMock())
    configurator.delete_configuration = MagicMock()

    class FakeConfiguration:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "bluepyemodel.efeatures_extraction.targets_configurator.TargetsConfiguration",
        FakeConfiguration,
    )
    configurator.new_configuration(
        files=["file"],
        targets=["target"],
        protocols_rheobase=["Rin"],
        auto_targets=None,
        protocols_mapping={"Rin": "Rin"},
    )

    configurator.delete_configuration.assert_called_once_with()
    assert configurator.configuration.kwargs["available_traces"] == ["trace"]
    assert configurator.configuration.kwargs["available_efeatures"] == ["feature"]


def test_targets_configurator_local_load_is_not_implemented():
    local_access_point = object.__new__(LocalAccessPoint)
    configurator = TargetsConfigurator(local_access_point)

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        configurator.load_configuration()


def test_targets_configurator_save_valid_and_invalid(monkeypatch):
    access_point = MagicMock()
    valid = SimpleNamespace(is_configuration_valid=True)
    configurator = TargetsConfigurator(access_point, valid)

    configurator.save_configuration()
    access_point.store_targets_configuration.assert_called_once_with(valid)

    configurator.configuration = SimpleNamespace(is_configuration_valid=False)
    with pytest.raises(ValueError, match="invalid configuration"):
        configurator.save_configuration()


def test_create_configuration_skips_existing_access_point():
    access_point = MagicMock()
    access_point.has_targets_configuration.return_value = True
    configurator = TargetsConfigurator(access_point)

    configurator.create_and_save_configuration_from_access_point()

    access_point.pipeline_settings.assert_not_called()


def test_create_configuration_uses_auto_targets(monkeypatch):
    access_point = MagicMock()
    access_point.has_targets_configuration.return_value = False
    access_point.pipeline_settings.files_for_extraction = ["file"]
    access_point.pipeline_settings.targets = []
    access_point.pipeline_settings.protocols_rheobase = ["Rin"]
    access_point.pipeline_settings.protocols_mapping = {"Rin": "Rin"}
    access_point.pipeline_settings.auto_targets = [{"preset": "direct"}]
    access_point.pipeline_settings.auto_targets_presets = []
    configurator = TargetsConfigurator(access_point)
    configurator.new_configuration = MagicMock()
    configurator.save_configuration = MagicMock()

    configurator.create_and_save_configuration_from_access_point()

    configurator.new_configuration.assert_called_once_with(
        ["file"], [], ["Rin"], [{"preset": "direct"}], {"Rin": "Rin"}
    )
    configurator.save_configuration.assert_called_once_with()


def test_create_configuration_uses_preset_auto_targets(monkeypatch):
    access_point = MagicMock()
    access_point.has_targets_configuration.return_value = False
    settings = access_point.pipeline_settings
    settings.files_for_extraction = []
    settings.targets = []
    settings.protocols_rheobase = []
    settings.protocols_mapping = None
    settings.auto_targets = None
    settings.auto_targets_presets = ["iv"]
    configurator = TargetsConfigurator(access_point)
    configurator.new_configuration = MagicMock()
    configurator.save_configuration = MagicMock()
    monkeypatch.setattr(
        "bluepyemodel.efeatures_extraction.targets_configurator.get_auto_target_from_presets",
        lambda presets: [{"preset": presets[0]}],
    )

    configurator.create_and_save_configuration_from_access_point()

    assert configurator.new_configuration.call_args.args[3] == [{"preset": "iv"}]


def test_create_configuration_requires_target_source():
    access_point = MagicMock()
    access_point.has_targets_configuration.return_value = False
    settings = access_point.pipeline_settings
    settings.targets = []
    settings.auto_targets = None
    settings.auto_targets_presets = []

    with pytest.raises(TypeError, match="either targets"):
        TargetsConfigurator(
            access_point
        ).create_and_save_configuration_from_access_point()
