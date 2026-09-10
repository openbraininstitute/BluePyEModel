"""Tests for MechanismConfiguration, DistributionConfiguration, ParameterConfiguration
and configure_model (bluepyemodel.model.*).
"""

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

from bluepyemodel.model.distribution_configuration import DistributionConfiguration
from bluepyemodel.model.mechanism_configuration import MechanismConfiguration
from bluepyemodel.model.model_configuration import configure_model
from bluepyemodel.model.parameter_configuration import ParameterConfiguration

# ---------------------------------------------------------------------------
# MechanismConfiguration
# ---------------------------------------------------------------------------


def test_mechanism_configuration_defaults():
    mech = MechanismConfiguration(name="NaTg", location="somatic")

    assert mech.name == "NaTg"
    assert mech.location == "somatic"
    assert mech.parameters == {}
    assert mech.ionic_concentrations == []
    assert mech.stochastic is False


def test_mechanism_configuration_stochastic_inferred_from_name():
    mech = MechanismConfiguration(name="StochKv3", location="somatic")
    assert mech.stochastic is True


def test_mechanism_configuration_ionic_concentrations_from_ion_currents():
    mech = MechanismConfiguration(name="Ca_HVA2", location="somatic", ion_currents=["ica"])
    assert mech.ionic_concentrations == ["cai"]


def test_mechanism_configuration_parameters_from_string():
    mech = MechanismConfiguration(name="pas", location="somatic", parameters="e")
    assert mech.parameters == {"e": [None, None]}


def test_mechanism_configuration_parameters_from_list_untouched():
    mech = MechanismConfiguration(name="pas", location="somatic", parameters={"e": [-90, -60]})
    assert mech.parameters == {"e": [-90, -60]}


def test_mechanism_configuration_get_current():
    mech = MechanismConfiguration(
        name="Ih",
        location="somatic",
        ion_currents=[],
        nonspecific_currents=["ihcn"],
    )
    assert mech.get_current() == ["ihcn_Ih"]


def test_mechanism_configuration_get_current_empty_when_none():
    mech = MechanismConfiguration(name="pas", location="somatic")
    assert mech.get_current() == []


def test_mechanism_configuration_as_dict():
    mech = MechanismConfiguration(
        name="NaTg", location="somatic", version="v1", temperature=34, id="nexus-id"
    )
    assert mech.as_dict() == {
        "name": "NaTg",
        "stochastic": False,
        "location": "somatic",
        "version": "v1",
        "temperature": 34,
        "ljp_corrected": None,
        "id": "nexus-id",
    }


# ---------------------------------------------------------------------------
# DistributionConfiguration
# ---------------------------------------------------------------------------


def test_distribution_configuration_defaults():
    distr = DistributionConfiguration(name="uniform")
    assert distr.parameters == []
    assert distr.morphology_dependent_parameters == []
    assert distr.soma_ref_location == 0.5


def test_distribution_configuration_soma_ref_location_none_defaults_to_half():
    distr = DistributionConfiguration(name="uniform", soma_ref_location=None)
    assert distr.soma_ref_location == 0.5


def test_distribution_configuration_parameters_from_string():
    distr = DistributionConfiguration(name="exp", parameters="constant")
    assert distr.parameters == ["constant"]


def test_distribution_configuration_morphology_dependent_parameters_from_string():
    distr = DistributionConfiguration(name="exp", morphology_dependent_parameters="length")
    assert distr.morphology_dependent_parameters == ["length"]


def test_distribution_configuration_as_dict_minimal():
    distr = DistributionConfiguration(name="uniform")
    assert distr.as_dict() == {
        "name": "uniform",
        "function": None,
        "soma_ref_location": 0.5,
    }


def test_distribution_configuration_as_dict_full():
    distr = DistributionConfiguration(
        name="exp",
        function="{value} * math.exp({distance})",
        parameters=["constant"],
        morphology_dependent_parameters=["length"],
        comment="a comment",
    )
    d = distr.as_dict()
    assert d["parameters"] == ["constant"]
    assert d["morphology_dependent_parameters"] == ["length"]
    assert d["comment"] == "a comment"


def test_distribution_configuration_as_legacy_dict():
    distr = DistributionConfiguration(
        name="exp", function="{value} * math.exp({distance})", parameters=["constant"]
    )
    assert distr.as_legacy_dict() == {
        "fun": "{value} * math.exp({distance})",
        "parameters": ["constant"],
    }


def test_distribution_configuration_as_legacy_dict_no_parameters():
    distr = DistributionConfiguration(name="uniform", function=None)
    assert distr.as_legacy_dict() == {"fun": None}


# ---------------------------------------------------------------------------
# ParameterConfiguration
# ---------------------------------------------------------------------------


def test_parameter_configuration_defaults():
    param = ParameterConfiguration(name="gNaTgbar", location="somatic", value=[0, 1])
    assert param.distribution == "uniform"
    assert param.valid_value is True


def test_parameter_configuration_distribution_none_defaults_to_uniform():
    param = ParameterConfiguration(
        name="gNaTgbar", location="somatic", value=0.1, distribution=None
    )
    assert param.distribution == "uniform"


def test_parameter_configuration_value_tuple_converted_to_list():
    param = ParameterConfiguration(name="gNaTgbar", location="somatic", value=(0, 1))
    assert param.value == [0, 1]


def test_parameter_configuration_single_element_list_becomes_scalar():
    param = ParameterConfiguration(name="gNaTgbar", location="somatic", value=[0.5])
    assert param.value == 0.5


def test_parameter_configuration_valid_value_false_when_none():
    param = ParameterConfiguration(name="gNaTgbar", location="somatic", value=None)
    assert param.valid_value is False


def test_parameter_configuration_as_dict_minimal():
    param = ParameterConfiguration(name="gNaTgbar", location="somatic", value=0.1)
    assert param.as_dict() == {"name": "gNaTgbar", "value": 0.1, "location": "somatic"}


def test_parameter_configuration_as_dict_with_distribution_and_mechanism():
    param = ParameterConfiguration(
        name="gNaTgbar", location="somatic", value=0.1, distribution="exp", mechanism="NaTg"
    )
    d = param.as_dict()
    assert d["distribution"] == "exp"
    assert d["mechanism"] == "NaTg"


def test_parameter_configuration_as_legacy_dict():
    param = ParameterConfiguration(
        name="gNaTgbar", location="somatic", value=0.1, distribution="exp"
    )
    assert param.as_legacy_dict() == {"name": "gNaTgbar", "val": 0.1, "dist": "exp"}


def test_parameter_configuration_as_legacy_dict_uniform_omits_dist():
    param = ParameterConfiguration(name="gNaTgbar", location="somatic", value=0.1)
    assert param.as_legacy_dict() == {"name": "gNaTgbar", "val": 0.1}


def test_parameter_configuration_eq_same_name_and_location():
    param1 = ParameterConfiguration(name="gNaTgbar", location="somatic", value=0.1)
    param2 = ParameterConfiguration(name="gNaTgbar", location="somatic", value=0.2)
    assert param1 == param2


def test_parameter_configuration_eq_name_all_ignores_location():
    # __eq__ treats a parameter named "all" as matching any location with the
    # same name, regardless of the actual `location` attribute value.
    param1 = ParameterConfiguration(name="all", location="somatic", value=0.1)
    param2 = ParameterConfiguration(name="all", location="axonal", value=0.2)
    assert param1 == param2


def test_parameter_configuration_eq_other_location_all():
    param1 = ParameterConfiguration(name="gNaTgbar", location="somatic", value=0.1)
    param2 = ParameterConfiguration(name="gNaTgbar", location="all", value=0.2)
    assert param1 == param2


def test_parameter_configuration_not_eq_different_name():
    param1 = ParameterConfiguration(name="gNaTgbar", location="somatic", value=0.1)
    param2 = ParameterConfiguration(name="gK_Tstbar", location="somatic", value=0.2)
    assert param1 != param2


def test_parameter_configuration_not_eq_different_location():
    param1 = ParameterConfiguration(name="gNaTgbar", location="somatic", value=0.1)
    param2 = ParameterConfiguration(name="gNaTgbar", location="axonal", value=0.2)
    assert param1 != param2


# ---------------------------------------------------------------------------
# configure_model
# ---------------------------------------------------------------------------


@patch("bluepyemodel.model.model_configuration.ModelConfigurator")
def test_configure_model(mock_configurator_cls):
    mock_configurator = MagicMock()
    mock_configurator_cls.return_value = mock_configurator
    access_point = MagicMock()

    result = configure_model(
        access_point,
        morphology_name="C060114A5",
        morphology_path="/tmp/C060114A5.asc",
        morphology_format="asc",
        use_gene_data=False,
    )

    mock_configurator_cls.assert_called_once_with(access_point=access_point)
    mock_configurator.new_configuration.assert_called_once_with(use_gene_data=False)
    mock_configurator.configuration.select_morphology.assert_called_once_with(
        "C060114A5",
        morphology_path="/tmp/C060114A5.asc",
        morphology_format="asc",
    )
    mock_configurator.save_configuration.assert_called_once()
    assert result is mock_configurator.configuration
