"""Deterministic tests for FitnessCalculatorConfiguration initialisation paths."""

import pytest

from bluepyemodel.evaluation.fitness_calculator_configuration import (
    FitnessCalculatorConfiguration,
)


def bluepyefe_protocol(amp=0.2, holding=-0.05):
    return {
        "step": {
            "delay": 250.0,
            "amp": amp,
            "thresh_perc": 130.0,
            "duration": 1350.0,
            "totduration": 1850.0,
        },
        "holding": {"amp": holding},
    }


def feature(name="mean_frequency", value=(6.0, 1.0), **extra):
    return {"feature": name, "val": list(value), **extra}


def test_init_from_bluepyefe_builds_protocols_and_features():
    configuration = FitnessCalculatorConfiguration(
        name_rmp_protocol="IDrest_100",
        name_rin_protocol="IV_-40",
        validation_protocols=["APWaveform_200"],
    )
    protocols = {
        "IDrest_100": bluepyefe_protocol(),
        "IV_-40": bluepyefe_protocol(amp=-0.04),
        "APWaveform_200": bluepyefe_protocol(amp=0.4),
    }
    efeatures = {
        "IDrest_100": {"soma": [feature("voltage_base", (-75.0, 2.0)), feature()]},
        "IV_-40": {
            "soma": [
                feature("voltage_base", (-80.0, 1.0)),
                feature("ohmic_input_resistance_vb_ssse", (150.0, 10.0)),
            ]
        },
        "APWaveform_200": {"soma": [feature("AP_amplitude", (70.0, 5.0))]},
    }
    currents = {"holding_current": [-0.05, 0.01], "threshold_current": [0.2, 0.02]}

    configuration.init_from_bluepyefe(efeatures, protocols, currents, None)

    protocol_names = [p.name for p in configuration.protocols]
    # IV_-40 is dropped because both of its efeatures are remapped to pre-protocols.
    assert set(protocol_names) == {"IDrest_100", "APWaveform_200"}
    assert all(
        p.protocol_type == "ThresholdBasedProtocol" for p in configuration.protocols
    )
    validation_flags = {p.name: p.validation for p in configuration.protocols}
    assert validation_flags["APWaveform_200"] is True
    assert validation_flags["IDrest_100"] is False

    by_protocol = {}
    for efeature in configuration.efeatures:
        by_protocol.setdefault(efeature.protocol_name, []).append(
            efeature.efel_feature_name
        )

    assert "steady_state_voltage_stimend" in by_protocol["RMPProtocol"]
    assert "steady_state_voltage_stimend" in by_protocol["SearchHoldingCurrent"]
    assert "ohmic_input_resistance_vb_ssse" in by_protocol["RinProtocol"]
    assert by_protocol["SearchThresholdCurrent"] == ["bpo_threshold_current"]
    holding = next(
        e
        for e in configuration.efeatures
        if e.efel_feature_name == "bpo_holding_current"
    )
    assert holding.mean == pytest.approx(-0.05)


def test_init_from_bluepyefe_applies_protocol_mapping():
    configuration = FitnessCalculatorConfiguration(
        validation_protocols=["raw_validation"]
    )
    protocols = {"raw_name": bluepyefe_protocol()}
    efeatures = {"raw_name": {"soma": [feature()]}}
    mapping = {"raw_name": "Step_100", "raw_validation": "Step_200"}

    configuration.init_from_bluepyefe(
        efeatures, protocols, None, None, protocols_mapping=mapping
    )

    assert [p.name for p in configuration.protocols] == ["Step_100"]
    assert configuration.validation_protocols == ["Step_200"]
    assert configuration.protocols[0].protocol_type == "Protocol"


def test_init_from_bluepyefe_requires_rmp_and_rin_stimuli():
    with pytest.raises(ValueError, match="requested for RMP"):
        FitnessCalculatorConfiguration(
            name_rmp_protocol="IDrest_100"
        ).init_from_bluepyefe({"IV_-40": {}}, {}, None, None)

    with pytest.raises(ValueError, match="requested for Rin"):
        FitnessCalculatorConfiguration(name_rin_protocol="IV_-40").init_from_bluepyefe(
            {"IDrest_100": {}}, {}, None, None
        )


def test_protocol_exist_is_always_truthy():
    """Documents current behaviour: protocol_exist wraps a generator, so it never returns False.

    As a consequence the "protocol does not exist" guards in _add_bluepyefe_efeature and
    _add_legacy_efeature are unreachable.
    """
    configuration = FitnessCalculatorConfiguration()

    assert configuration.protocol_exist("never_registered") is True

    configuration._add_bluepyefe_efeature(feature(), "Unknown_100", "soma", None)
    assert configuration.efeatures[0].protocol_name == "Unknown_100"


def test_bluepyefe_efeature_uses_optional_metadata():
    configuration = FitnessCalculatorConfiguration()
    configuration._add_bluepyefe_protocol("IDrest_100", bluepyefe_protocol())
    configuration._add_bluepyefe_efeature(
        feature(
            efeature_name="custom",
            n=12,
            weight=3.0,
            efel_settings={"strict_stiminterval": True},
        ),
        "IDrest_100",
        "dend.v",
        None,
    )

    efeature = configuration.efeatures[0]
    assert efeature.recording_name == "dend.v"
    assert efeature.efeature_name == "custom"
    assert efeature.sample_size == 12
    assert efeature.weight == pytest.approx(3.0)
    assert efeature.efel_settings["strict_stiminterval"] is True


def legacy_protocol(with_holding=True, protocol_type=None, extra_recordings=None):
    protocol = {
        "stimuli": {
            "step": {
                "delay": 250.0,
                "amp": 0.2,
                "duration": 1350.0,
                "totduration": 1850.0,
            }
        }
    }
    if with_holding:
        protocol["stimuli"]["holding"] = {"amp": -0.05}
    if protocol_type:
        protocol["type"] = protocol_type
    if extra_recordings:
        protocol["extra_recordings"] = extra_recordings
    return protocol


def test_init_from_legacy_dict_builds_pre_protocols_and_features():
    configuration = FitnessCalculatorConfiguration()
    protocols = {
        "Step_150": legacy_protocol(protocol_type="StepThresholdProtocol"),
        "RMP": {"stimuli": {"step": {"duration": 500.0}}},
        "Rin": {
            "stimuli": {
                "step": {
                    "delay": 100.0,
                    "duration": 400.0,
                    "amp": -0.02,
                    "totduration": 600.0,
                }
            }
        },
        "ThresholdDetection": {
            "step_template": {
                "stimuli": {
                    "step": {"delay": 100.0, "duration": 300.0, "totduration": 500.0}
                }
            }
        },
    }
    efeatures = {
        "Step_150": {"soma": [feature()]},
        "RMP": {
            "soma": [
                feature("voltage_base", (-75.0, 2.0)),
                feature("Spikecount", (0.0, 1.0)),
            ]
        },
        "Rin": {
            "soma": [
                feature("ohmic_input_resistance_vb_ssse", (150.0, 10.0)),
                feature("voltage_base", (-80.0, 1.0)),
            ]
        },
        "RinHoldCurrent": {"soma": [feature("bpo_holding_current", (-0.05, 0.01))]},
        "Threshold": {"soma": [feature("bpo_threshold_current", (0.2, 0.02))]},
    }

    configuration.init_from_legacy_dict(efeatures, protocols, None)

    assert [p.name for p in configuration.protocols] == ["Step_150"]
    assert configuration.protocols[0].protocol_type == "ThresholdBasedProtocol"
    assert configuration.rmp_duration == 500.0
    assert configuration.rin_step_delay == 100.0
    assert configuration.rin_step_amp == pytest.approx(-0.02)
    assert configuration.search_threshold_step_duration == 300.0

    protocol_names = {e.protocol_name for e in configuration.efeatures}
    assert "RMPProtocol" in protocol_names
    assert "RinProtocol" in protocol_names
    assert "SearchHoldingCurrent" in protocol_names
    assert "SearchThresholdCurrent" in protocol_names


def test_legacy_protocol_without_holding_and_with_extra_recordings():
    configuration = FitnessCalculatorConfiguration()
    configuration._add_legacy_protocol(
        "Step_150",
        legacy_protocol(
            with_holding=False,
            extra_recordings=[
                {
                    "name": "dend",
                    "var": "v",
                    "type": "somadistance",
                    "somadistance": 100,
                }
            ],
        ),
    )

    protocol = configuration.protocols[0]
    assert protocol.stimuli[0]["holding_current"] is None
    assert protocol.protocol_type == "Protocol"
    assert any("Step_150.dend.v" == rec["name"] for rec in protocol.recordings)


def test_legacy_efeature_skips_unrelated_pre_protocol_features():
    configuration = FitnessCalculatorConfiguration()

    configuration._add_legacy_efeature(
        feature("AP_amplitude", (70.0, 5.0)), "Rin", "soma", None
    )
    configuration._add_legacy_efeature(
        feature("AP_amplitude", (70.0, 5.0)), "RMP", "soma", None
    )

    assert configuration.efeatures == []


def test_legacy_efeature_maps_pre_protocol_names():
    configuration = FitnessCalculatorConfiguration()

    configuration._add_legacy_efeature(
        feature("bpo_holding_current", (-0.05, 0.01)), "RinHoldCurrent", "soma", None
    )
    configuration._add_legacy_efeature(
        feature("bpo_threshold_current", (0.2, 0.02)), "Threshold", "soma", None
    )

    assert [e.protocol_name for e in configuration.efeatures] == [
        "SearchHoldingCurrent",
        "SearchThresholdCurrent",
    ]


def test_init_from_legacy_dict_detects_validation_mismatch():
    configuration = FitnessCalculatorConfiguration()
    protocols = {"Step_150": {**legacy_protocol(), "validation": True}}

    with pytest.raises(ValueError, match="validation protocol"):
        configuration.init_from_legacy_dict(
            {"Step_150": {"soma": [feature()]}}, protocols, None
        )


def test_initialise_efeatures_respects_per_feature_overrides():
    configuration = FitnessCalculatorConfiguration()
    efeatures = [
        {
            "efel_feature_name": "mean_frequency",
            "protocol_name": "Step_150",
            "recording_name": "soma.v",
            "mean": 6.0,
            "std": 0.0,
            "threshold_efeature_std": 0.5,
            "default_std_value": 0.2,
        }
    ]

    configured = configuration.initialise_efeatures(efeatures)

    assert configured[0].threshold_efeature_std == pytest.approx(0.5)


def test_initialise_protocols_creates_configurations():
    configuration = FitnessCalculatorConfiguration()
    protocols = [
        {
            "name": "Step_150",
            "stimuli": [
                {"delay": 250.0, "amp": 0.2, "duration": 1350.0, "totduration": 1850.0}
            ],
            "recordings_from_config": [
                {
                    "type": "CompRecording",
                    "name": "Step_150.soma.v",
                    "location": "soma",
                    "variable": "v",
                }
            ],
        }
    ]

    configured = configuration.initialise_protocols(protocols)

    assert [p.name for p in configured] == ["Step_150"]
