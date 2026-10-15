"""Deterministic model configuration, model factory, and eCode coverage."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bluepyopt import ephys
from bluepyopt.ephys.parameterscalers import (
    NrnSegmentLinearScaler,
    NrnSegmentSomaDistanceScaler,
    NrnSegmentSomaDistanceStepScaler,
)

from bluepyemodel.ecode import eCodes, fixed_timestep_eCodes
from bluepyemodel.ecode.hyperdepol import HyperDepol
from bluepyemodel.ecode.sahp import sAHP
from bluepyemodel.ecode.spikerec import SpikeRecMultiSpikes
from bluepyemodel.ecode.stimulus import BPEM_stimulus
from bluepyemodel.model import model
from bluepyemodel.model.distribution_configuration import DistributionConfiguration
from bluepyemodel.model.mechanism_configuration import MechanismConfiguration
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration
from bluepyemodel.model.parameter_configuration import ParameterConfiguration


def _available_mechanisms():
    return [
        MechanismConfiguration(
            name="test_mechanism", location=None, parameters={"gbar": [0, 1]}
        ),
        MechanismConfiguration(name="test_mechanism2", location=None),
    ]


def _configuration():
    return NeuronModelConfiguration(
        available_mechanisms=_available_mechanisms(),
        available_morphologies=["M1"],
        extra_mech_ids=[("extra-id", "ExtraType")],
    )


def test_configuration_constructor_properties_and_location_formatting():
    configuration = NeuronModelConfiguration(
        parameters={
            "name": "g",
            "location": "soma",
            "value": 0.1,
            "distribution": "exp",
        },
        mechanisms={"name": "pas", "location": "soma"},
        distributions={"name": "exp", "function": "{value}"},
        morphology={"name": "M1", "path": "/tmp/M1.asc", "format": "asc"},
    )

    assert configuration.mechanism_names == {"pas"}
    assert configuration.distribution_names == {"exp", "uniform"}
    assert configuration.used_distribution_names == {"exp"}
    assert NeuronModelConfiguration._format_locations(None) == []
    assert NeuronModelConfiguration._format_locations("soma") == ["soma"]
    locations = ["soma", "basal"]
    assert NeuronModelConfiguration._format_locations(locations) is locations


def test_configuration_add_distribution_parameter_and_duplicate_warning(caplog):
    configuration = _configuration()
    configuration.add_distribution(
        "exp", "{value} * {distance}", parameters=["distance"]
    )
    configuration.add_distribution("exp", "{value}")
    assert len(configuration.distributions) == 3
    assert "Distribution exp already exists" in caplog.text

    configuration.add_parameter("g", ["soma", "basal"], [0, 1], distribution_name="exp")
    configuration.add_parameter("g", "soma", [0, 1], distribution_name="constant")
    assert len(configuration.parameters) == 2

    with pytest.raises(ValueError, match="without specifying a location"):
        configuration.add_parameter("missing", None, 1)
    with pytest.raises(ValueError, match="register your distributions"):
        configuration.add_parameter("missing", "soma", 1, distribution_name="missing")


def test_configuration_mechanism_availability_and_auto_parameters():
    configuration = _configuration()
    assert configuration.is_mechanism_available("test_mechanism")
    assert not configuration.is_mechanism_available("unknown")
    assert NeuronModelConfiguration().is_mechanism_available("anything")

    configuration.add_mechanism(
        "test_mechanism", "soma", auto_parameter=True, id="mech-id"
    )
    assert configuration.parameters[0].name == "gbar"
    assert configuration.parameters[0].value is None
    assert configuration.mechanisms[0].id == "mech-id"

    configuration.add_mechanism("test_mechanism", "soma")
    assert len(configuration.mechanisms) == 1
    configuration.add_mechanism("pas", "somatic")
    assert {mechanism.name for mechanism in configuration.mechanisms} == {
        "test_mechanism",
        "pas",
    }

    with pytest.raises(ValueError, match="not available"):
        configuration.add_mechanism("unknown", "soma")


def test_configuration_init_from_dict_and_serialization(caplog):
    configuration = _configuration()
    configuration.init_from_dict(
        {
            "distributions": [{"name": "exp", "function": "{value} * {distance}"}],
            "parameters": [
                {
                    "name": "g",
                    "location": "soma",
                    "value": [0, 1],
                    "dist": "exp",
                }
            ],
            "mechanisms": [{"name": "pas", "location": "somatic", "id": "pas-id"}],
            "morphology": {"name": "M1", "path": "/tmp/M1.asc", "format": "asc"},
        },
        {},
    )

    assert configuration.morphology.name == "M1"
    assert configuration.parameters[0].distribution == "exp"
    assert configuration.mechanisms[0].id == "pas-id"
    assert configuration.as_dict()["distributions"][0]["name"] == "exp"
    assert "Mechanisms:" in str(configuration)
    assert "Removing" in caplog.text


def test_configuration_legacy_dict_and_related_ids():
    configuration = _configuration()
    configuration.init_from_legacy_dict(
        {
            "distributions": {"exp": {"fun": "{value} * {distance}"}},
            "parameters": {
                "__comment": [],
                "soma": [{"name": "g_test_mechanism", "val": [0, 1], "dist": "exp"}],
                "distribution_exp": [{"name": "constant", "val": [1, 2]}],
                "global": [{"name": "celsius", "val": 34}],
                "soma_ion": [{"name": "cai_ion", "val": 0.1}],
            },
            "mechanisms": {"soma": {"mech": ["test_mechanism"]}},
        },
        {"name": "M1", "path": "/tmp/M1.asc", "format": "asc", "id": "m-id"},
    )

    configuration.mechanisms.append(
        MechanismConfiguration(name="duplicate", location="soma", id="mech-id")
    )
    configuration.mechanisms.append(
        MechanismConfiguration(name="duplicate2", location="soma", id="mech-id")
    )
    assert configuration.get_related_nexus_ids() == {
        "uses": [
            {"id": "m-id", "type": "NeuronMorphology"},
            {"id": "mech-id", "type": "SubCellularModelScript"},
            {"id": "extra-id", "type": "ExtraType"},
        ]
    }

    with pytest.raises(ValueError, match="Could not find mechanism"):
        configuration.init_from_legacy_dict(
            {
                "distributions": {},
                "parameters": {"soma": [{"name": "unknown", "val": 1}]},
                "mechanisms": {},
            },
            {"name": "M1"},
        )


def test_configuration_set_remove_and_morphology_selection():
    configuration = _configuration()
    configuration.select_morphology(
        "M1",
        morphology_path="/tmp/M1.asc",
        morphology_format="asc",
        seclist_names=["somatic"],
        secarray_names=["soma"],
        section_index=2,
    )
    configuration.add_parameter("g", ["soma", "basal"], [0, 1])
    configuration.set_parameter_distribution("g", "soma", "uniform")
    configuration.set_parameter_value("g", "soma", 0.25)
    assert configuration.parameters[0].value == 0.25

    with pytest.raises(ValueError, match="without specifying"):
        configuration.set_parameter_value("g", None, 1)
    with pytest.raises(ValueError, match="not available"):
        configuration.select_morphology("missing")

    configuration.add_mechanism("test_mechanism", "soma")
    configuration.remove_parameter("g", ["soma"])
    configuration.remove_mechanism("test_mechanism", "soma")
    assert all(parameter.location != "soma" for parameter in configuration.parameters)
    configuration.remove_parameter("g")
    assert not configuration.parameters


def test_model_location_distribution_and_parameter_branches(caplog, monkeypatch):
    assert len(model.multi_locations("somadend", {})) == 3
    assert len(model.multi_locations("unknown", {})) == 1
    assert len(model.multi_locations("custom", {"custom": ["basal"]})) == 1
    model.multi_locations("custom", {"custom": ["unexpected"]})
    assert "not in expected locations" in caplog.text

    definitions = [
        DistributionConfiguration("uniform"),
        DistributionConfiguration("exp", "{value} * {distance}", ["distance"]),
        DistributionConfiguration("step", "{value} * {distance}", ["distance"]),
    ]
    monkeypatch.setattr(
        "bluepyemodel.model.model.get_hotspot_location", lambda _: (1.0, 2.0)
    )
    distributions = model.define_distributions(
        definitions, SimpleNamespace(morphology_path="M1")
    )
    assert isinstance(distributions["uniform"], NrnSegmentLinearScaler)
    assert isinstance(distributions["exp"], NrnSegmentSomaDistanceScaler)
    assert isinstance(distributions["step"], NrnSegmentSomaDistanceStepScaler)

    parameter_definitions = [
        ParameterConfiguration(name="celsius", location="global", value=34),
        ParameterConfiguration(name="g", location="distribution_exp", value=[0, 1]),
        ParameterConfiguration(name="h", location="soma", value=0.2),
    ]
    parameters = model.define_parameters(parameter_definitions, distributions, {})
    assert isinstance(parameters[0], ephys.parameters.NrnGlobalParameter)
    assert isinstance(parameters[1], ephys.parameters.MetaParameter)
    assert isinstance(parameters[2], ephys.parameters.NrnSectionParameter)
    with pytest.raises(ValueError, match="Lower bound"):
        model.define_parameters(
            [ParameterConfiguration(name="bad", location="soma", value=[2, 1])],
            distributions,
            {},
        )


def test_model_mechanism_morphology_and_attribute_branches(monkeypatch):
    mechanisms = model.define_mechanisms(
        [MechanismConfiguration(name="pas", location="somadend", stochastic=True)], {}
    )
    assert mechanisms[0].name == "pas.somadend"
    assert mechanisms[0].deterministic is False

    morphology = SimpleNamespace(path="/tmp/M1.asc")
    configuration = SimpleNamespace(morphology=morphology)
    fake_morphology = MagicMock()
    monkeypatch.setattr("bluepyemodel.model.model.NrnFileMorphology", fake_morphology)

    model.define_morphology(configuration, morph_modifiers=None)
    assert fake_morphology.call_args.kwargs["morph_modifiers"]
    model.define_morphology(configuration, morph_modifiers=[])
    model.define_morphology(configuration, morph_modifiers=["bluepyopt_replace_axon"])
    assert fake_morphology.call_args.kwargs["do_replace_axon"] is True
    model.define_morphology(configuration, morph_modifiers=[lambda _: None])

    with pytest.raises(ValueError, match="Invalid morph_modifier"):
        model.define_morphology(configuration, morph_modifiers=[["only-one"]])
    with pytest.raises(TypeError, match="not callable"):
        model.define_morphology(configuration, morph_modifiers=[42])

    module = SimpleNamespace(foo=3, foo_hoc="hoc")
    assert model._get_attribute(module, "foo") == 3
    assert model._get_attribute(module, "foo", default_hoc=True) == 3
    assert (
        model._get_attribute(SimpleNamespace(foo_hoc="hoc"), "foo", default_hoc=True)
        == "hoc"
    )
    assert model._get_attribute(None, "foo") is None


def test_stimulus_defaults_and_destroy():
    stimulus = BPEM_stimulus(None)
    assert stimulus.stim_start == stimulus.stim_end == stimulus.amplitude == 0.0
    assert stimulus.generate(dt=1) == ([], [])
    assert str(stimulus) == " current played at None"
    stimulus.current_vec = object()
    stimulus.time_vec = object()
    stimulus.destroy()
    assert stimulus.current_vec is None
    assert stimulus.time_vec is None


def test_relative_amplitude_ecodes():
    hyperdepol = HyperDepol(None, hyper_amp_rel=-100, depol_amp_rel=200, totduration=10)
    hyperdepol.threshold_current = 0.2
    assert hyperdepol.amplitude == -0.2
    assert hyperdepol.depol_amplitude == 0.4

    ahp = sAHP(None, thresh_perc=150, long_amp_rel=40)
    ahp.threshold_current = 0.2
    assert ahp.amplitude == pytest.approx(0.3)
    assert ahp.long_amplitude == pytest.approx(0.08)

    spikes = SpikeRecMultiSpikes(
        None, thresh_perc=50, n_spikes=3, delay=1, spike_duration=2, delta=3
    )
    spikes.threshold_current = 0.2
    assert spikes.amplitude == 0.1
    assert spikes.multi_stim_start() == [1, 6, 11]
    assert spikes.multi_stim_end() == [3, 8, 13]
    assert spikes.stim_end == 13

    with pytest.raises(TypeError, match="hyper_amp"):
        HyperDepol(None, depol_amp=1)
    with pytest.raises(TypeError, match="depol_amp"):
        HyperDepol(None, hyper_amp=1)
    with pytest.raises(TypeError, match="amp and thresh_perc"):
        sAHP(None, long_amp=1)
    with pytest.raises(TypeError, match="long_amp"):
        sAHP(None, amp=1)


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("spontaneous", "IDrest"),
        ("step", "IDrest"),
        ("idthres", "IDrest"),
        ("idthresh", "IDrest"),
        ("bap", "IDrest"),
        ("spikerec", "IDrest"),
        ("startnohold", "IDrest"),
        ("ap_thresh", "Ramp"),
        ("apthresh", "Ramp"),
    ],
)
def test_ecode_registry_aliases(alias, expected):
    assert eCodes[alias].__name__ == expected
    assert fixed_timestep_eCodes == ["probampanmda_ems"]
