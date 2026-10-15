"""Tests for bluepyemodel.model.model_configurator.ModelConfigurator."""

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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.model.mechanism_configuration import MechanismConfiguration
from bluepyemodel.model.model_configurator import ModelConfigurator


@pytest.fixture
def mock_access_point():
    access_point = MagicMock()
    access_point.get_distributions.return_value = []
    access_point.get_available_mechanisms.return_value = []
    access_point.get_available_morphologies.return_value = ["C060114A5"]
    access_point.pipeline_settings.name_gene_map = None
    return access_point


@pytest.fixture
def mock_access_point_with_natg(mock_access_point):
    mock_access_point.get_available_mechanisms.return_value = [
        MechanismConfiguration(name="NaTg", location="somatic", temperature=34, ljp_corrected=True)
    ]
    return mock_access_point


@pytest.fixture
def mock_access_point_with_natg_versioned(mock_access_point):
    mock_access_point.get_available_mechanisms.return_value = [
        MechanismConfiguration(
            name="NaTg",
            location="somatic",
            version="sscx",
            temperature=34,
            ljp_corrected=True,
        )
    ]
    return mock_access_point


def test_new_configuration_without_gene_data(mock_access_point):
    configurator = ModelConfigurator(access_point=mock_access_point)

    configurator.new_configuration(use_gene_data=False)

    assert configurator.configuration is not None
    mock_access_point.get_distributions.assert_called_once()
    mock_access_point.get_available_mechanisms.assert_called_once()
    mock_access_point.get_available_morphologies.assert_called_once()


def test_new_configuration_deletes_existing_configuration(mock_access_point):
    configurator = ModelConfigurator(access_point=mock_access_point)
    configurator.new_configuration(use_gene_data=False)
    existing_configuration = configurator.configuration

    with patch("bluepyemodel.model.model_configurator.yesno", return_value=False):
        configurator.new_configuration(use_gene_data=False)

    assert configurator.configuration is not existing_configuration


def test_load_configuration_raises_for_local_access_point(mock_access_point):
    local_access_point = MagicMock(spec=LocalAccessPoint)
    configurator = ModelConfigurator(access_point=local_access_point)

    with pytest.raises(NotImplementedError):
        configurator.load_configuration()


def test_load_configuration_uses_access_point_for_non_local(mock_access_point):
    mock_access_point.get_model_configuration.return_value = "the_configuration"
    configurator = ModelConfigurator(access_point=mock_access_point)

    configurator.load_configuration()

    assert configurator.configuration == "the_configuration"


def test_save_configuration_stores_via_access_point(mock_access_point):
    configurator = ModelConfigurator(access_point=mock_access_point)
    configurator.new_configuration(use_gene_data=False)

    configurator.save_configuration(path="some/path")

    mock_access_point.store_model_configuration.assert_called_once_with(
        configurator.configuration, "some/path"
    )


def test_save_configuration_noop_without_configuration(mock_access_point):
    configurator = ModelConfigurator(access_point=mock_access_point)

    configurator.save_configuration()

    mock_access_point.store_model_configuration.assert_not_called()


def test_delete_configuration_saves_when_confirmed(mock_access_point):
    configurator = ModelConfigurator(access_point=mock_access_point)
    configurator.new_configuration(use_gene_data=False)

    with patch("bluepyemodel.model.model_configurator.yesno", return_value=True):
        configurator.delete_configuration()

    mock_access_point.store_model_configuration.assert_called_once()
    assert configurator.configuration is None


def test_delete_configuration_skips_save_when_declined(mock_access_point):
    configurator = ModelConfigurator(access_point=mock_access_point)
    configurator.new_configuration(use_gene_data=False)

    with patch("bluepyemodel.model.model_configurator.yesno", return_value=False):
        configurator.delete_configuration()

    mock_access_point.store_model_configuration.assert_not_called()
    assert configurator.configuration is None


def test_delete_configuration_noop_without_configuration(mock_access_point):
    configurator = ModelConfigurator(access_point=mock_access_point)

    configurator.delete_configuration()

    assert configurator.configuration is None


def test_get_gene_based_parameters_warns_without_gene_map(mock_access_point, caplog):
    configurator = ModelConfigurator(access_point=mock_access_point)

    parameters, mechanisms, distributions, nexus_keys = configurator.get_gene_based_parameters()

    assert parameters == []
    assert mechanisms == []
    assert distributions == []
    assert nexus_keys == []
    assert "No gene mapping name informed" in caplog.text


def test_get_gene_based_configuration_uses_icselector(mock_access_point_with_natg_versioned):
    mock_access_point = mock_access_point_with_natg_versioned
    mock_access_point.pipeline_settings.name_gene_map = "test_gene_map"
    mock_access_point.load_channel_gene_expression.return_value = (None, "gene_map_path")
    mock_access_point.load_ic_map.return_value = "ic_map_path"
    mock_access_point.emodel_metadata.mtype = "L5_TPC"
    mock_access_point.emodel_metadata.etype = "cAC"
    mock_access_point.emodel_metadata.ttype = "245_L5_PT_CTX"

    fake_selector = MagicMock()
    fake_selector.get_cell_config_from_ttype.return_value = (
        [{"name": "gNaTgbar_NaTg", "location": "somatic", "value": (0, 1)}],
        [
            {
                "name": "NaTg",
                "location": "somatic",
                "stochastic": False,
                "temperature": {"value": 34},
                "isLjpCorrected": True,
            }
        ],
        [{"name": "exp", "function": "{value}*{distance}", "parameters": ["constant"]}],
        [{"name": "NaTg", "modelId": "sscx"}],
    )

    configurator = ModelConfigurator(access_point=mock_access_point)

    with patch("bluepyemodel.icselector.icselector.ICSelector", return_value=fake_selector):
        configurator.get_gene_based_configuration()

    assert configurator.configuration is not None
    # "exp" from the gene-based config, plus the default "uniform" distribution
    # that NeuronModelConfiguration always adds.
    assert len(configurator.configuration.distributions) == 2
    assert {d.name for d in configurator.configuration.distributions} == {"exp", "uniform"}
    assert len(configurator.configuration.parameters) == 1
    assert len(configurator.configuration.mechanisms) == 1


def test_get_gene_based_configuration_uniform_distribution_skipped(mock_access_point_with_natg):
    mock_access_point = mock_access_point_with_natg
    mock_access_point.pipeline_settings.name_gene_map = "test_gene_map"
    mock_access_point.load_channel_gene_expression.return_value = (None, "gene_map_path")
    mock_access_point.load_ic_map.return_value = "ic_map_path"
    mock_access_point.emodel_metadata.mtype = "L5_TPC"
    mock_access_point.emodel_metadata.etype = "cAC"
    mock_access_point.emodel_metadata.ttype = "245_L5_PT_CTX"

    fake_selector = MagicMock()
    fake_selector.get_cell_config_from_ttype.return_value = (
        [],
        [],
        [{"name": "uniform"}, {"name": "constant"}],
        [],
    )

    configurator = ModelConfigurator(access_point=mock_access_point)

    with patch("bluepyemodel.icselector.icselector.ICSelector", return_value=fake_selector):
        configurator.get_gene_based_configuration()

    # "uniform" and "constant" distributions from the gene-based config should
    # be skipped: only the default "uniform" distribution auto-added by
    # NeuronModelConfiguration's constructor remains.
    assert len(configurator.configuration.distributions) == 1
    assert configurator.configuration.distributions[0].name == "uniform"


def test_get_gene_based_configuration_mechanism_uses_ljp_corrected_alias(
    mock_access_point_with_natg,
):
    mock_access_point = mock_access_point_with_natg
    mock_access_point.pipeline_settings.name_gene_map = "test_gene_map"
    mock_access_point.load_channel_gene_expression.return_value = (None, "gene_map_path")
    mock_access_point.load_ic_map.return_value = "ic_map_path"
    mock_access_point.emodel_metadata.mtype = "L5_TPC"
    mock_access_point.emodel_metadata.etype = "cAC"
    mock_access_point.emodel_metadata.ttype = "245_L5_PT_CTX"

    fake_selector = MagicMock()
    fake_selector.get_cell_config_from_ttype.return_value = (
        [],
        [
            {
                "name": "NaTg",
                "location": "somatic",
                "isLjpCorrected": True,
            }
        ],
        [],
        [],
    )

    configurator = ModelConfigurator(access_point=mock_access_point)

    with patch("bluepyemodel.icselector.icselector.ICSelector", return_value=fake_selector):
        configurator.get_gene_based_configuration()

    mech = configurator.configuration.mechanisms[0]
    assert mech.ljp_corrected is True
