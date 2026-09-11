"""Tests for extraction and evaluation configuration helpers."""

import pytest

from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration
from bluepyemodel.efeatures_extraction.trace_file import (
    TraceFile,
    list_ecodes_per_traces,
)
from bluepyemodel.evaluation.efeature_configuration import EFeatureConfiguration
from bluepyemodel.evaluation.protocol_configuration import ProtocolConfiguration
from bluepyemodel.evaluation.utils import (
    define_bAP_feature,
    define_bAP_protocol,
    define_EPSP_feature,
)


def test_trace_file_defaults_equality_and_ecode_counts():
    first = TraceFile(
        cell_name="cell_a",
        ecodes={"IDRest": {}, "IV": {}},
        id="trace-1",
    )
    same = TraceFile(cell_name="cell_a", filename="cell_a", ecodes={"IDRest": {}})
    other = TraceFile(cell_name="cell_b", filename="cell_b")

    assert first.filename == "cell_a"
    assert first == same
    assert first != other
    counts = list_ecodes_per_traces([first, other], threshold_count=0)
    assert set(counts["cell_a"]) == {"IDRest", "IV"}
    assert counts["cell_b"] == []
    assert set(counts["all"]) == {"IDRest", "IV"}
    counts = list_ecodes_per_traces([first, other], threshold_count=1)
    assert counts["cell_a"] == []
    assert counts["cell_b"] == []
    assert counts["all"] == []


def test_targets_configuration_validates_available_content():
    trace = TraceFile(cell_name="cell_a", filename="trace.abf", ecodes={"IDRest": {}})
    target = {
        "efeature": "Spikecount",
        "protocol": "IDRest",
        "amplitude": 150.0,
        "tolerance": 10.0,
    }
    configuration = TargetsConfiguration(
        files=[trace.as_dict()],
        targets=[target],
        protocols_rheobase="IDRest",
        available_traces=[trace],
        available_efeatures=["Spikecount"],
    )

    assert configuration.protocols_rheobase == ["IDRest"]
    assert configuration.is_configuration_valid is True
    assert configuration.protocols_rheobase_BPE == ["IDRest"]
    assert configuration.get_related_nexus_ids() == {"uses": []}


def test_targets_configuration_rejects_unavailable_target():
    trace = TraceFile(cell_name="cell_a", ecodes={"IDRest": {}})

    with pytest.raises(ValueError, match="Efeature name Spikecount does not exist"):
        TargetsConfiguration(
            files=[trace.as_dict()],
            targets=[
                {
                    "efeature": "Spikecount",
                    "protocol": "IDRest",
                    "amplitude": 150.0,
                    "tolerance": 10.0,
                }
            ],
            available_efeatures=["voltage_base"],
        )


def test_targets_configuration_auto_targets_and_presence_checks():
    configuration = TargetsConfiguration(
        files=[],
        auto_targets=[
            {
                "protocols": ["RMP", "Rin"],
                "amplitudes": [0, 100],
                "efeatures": [
                    "voltage_base",
                    "ohmic_input_resistance_vb_ssse",
                ],
            }
        ],
        protocols_rheobase=["RMP"],
    )

    assert configuration.is_configuration_valid is True
    assert configuration.targets_BPE is None
    assert configuration.auto_targets_BPE[0].protocols == ["RMP", "Rin"]
    configuration.check_presence_RMP_Rin_efeatures("RMP_0", "Rin_100")


def test_efeature_configuration_name_recording_and_std_branches():
    feature = EFeatureConfiguration(
        efel_feature_name="AP_amplitude",
        efeature_name="custom",
        protocol_name="Step_100",
        recording_name={"": "soma.v", "dend": "dend.v"},
        mean=0.0,
        original_std=0.0,
        threshold_efeature_std=0.2,
    )

    assert feature.name == "Step_100.soma.v.custom"
    assert feature.recording_name_for_instantiation == {
        "": "Step_100.soma.v",
        "dend": "Step_100.dend.v",
    }
    assert feature.std == 0.2
    assert feature.as_dict()["weight"] == 1.0


def test_protocol_configuration_expands_ion_recordings():
    recording = {
        "type": "CompRecording",
        "name": "Step_100.soma.v",
        "location": "soma",
        "variable": "v",
    }
    configuration = ProtocolConfiguration(
        name="Step_100",
        stimuli={"amp": 0.1},
        recordings_from_config=recording,
        ion_variables=["ina", "cai"],
        protocol_type="Protocol",
    )

    assert len(configuration.stimuli) == 1
    assert [r["name"] for r in configuration.recordings] == [
        "Step_100.soma.v",
        "Step_100.soma.ina",
        "Step_100.soma.cai",
    ]
    assert "recordings" not in configuration.as_dict()


def test_evaluation_helper_factories_create_expected_objects():
    protocol = define_bAP_protocol(
        dist_start=10, dist_end=30, dist_step=10, dist_end_basal=20
    )
    feature = define_bAP_feature(dist_start=10, dist_end=30, dist_step=10)
    epsp_feature = define_EPSP_feature(dist_start=100, dist_end=300, dist_step=100)

    assert protocol.name == "bAP_1000"
    assert len(protocol.recordings) == 1 + 2 + 1
    assert feature.distances == [0, 10, 20]
    assert epsp_feature.distances == [0, 100, 200]


def test_efeature_configuration_std_defaults_and_threshold_branches():
    legacy = EFeatureConfiguration("feature", "Step", "soma.v", 2.0, std=0.4)
    assert legacy.original_std == 0.4
    assert legacy.efel_settings == {"strict_stiminterval": True}
    assert legacy.std == 0.4

    limited = EFeatureConfiguration(
        "feature", "Step", "soma.v", 2.0, original_std=0.1, threshold_efeature_std=0.2
    )
    assert limited.std == 0.4

    zero_default = EFeatureConfiguration(
        "feature", "Step", "soma.v", 0.0, original_std=0.4, threshold_efeature_std=0
    )
    assert zero_default.std == 1e-3

    supplied = EFeatureConfiguration(
        "feature",
        "Step",
        "soma.v",
        0.0,
        original_std=0.4,
        threshold_efeature_std=0.2,
        efel_settings={"strict_stiminterval": False},
    )
    assert supplied.std == 0.2
    assert supplied.efel_settings == {"strict_stiminterval": False}


def test_protocol_configuration_backward_compatibility_and_normalization():
    recording = {"type": "CompRecording", "name": "Step.soma.v", "variable": "v"}
    configuration = ProtocolConfiguration(
        name="Step",
        stimuli={"amp": 0.1},
        recordings=recording,
        protocol_type="Protocol",
    )

    assert configuration.stimuli == [{"amp": 0.1}]
    assert configuration.recordings_from_config == [recording]
    assert configuration.as_dict()["recordings_from_config"] == [recording]

    with pytest.raises(ValueError, match="recordings_from_config"):
        ProtocolConfiguration(name="Step", stimuli=[], recordings_from_config=None)


def test_protocol_configuration_expands_legacy_var_and_skips_stimulus_recordings():
    compartment = {"type": "CompRecording", "name": "Step.soma.v", "var": "v"}
    stimulus = {
        "type": "FixedDtRecordingStimulus",
        "name": "Step.iclamp.i",
        "variable": "i",
    }
    configuration = ProtocolConfiguration(
        name="Step",
        stimuli=[],
        recordings_from_config=[compartment, stimulus],
        ion_variables=["ina", "cai"],
    )

    assert [recording["name"] for recording in configuration.recordings] == [
        "Step.soma.v",
        "Step.soma.ina",
        "Step.soma.cai",
        "Step.iclamp.i",
    ]
    assert configuration.recordings[1]["var"] == "ina"


def test_protocol_configuration_requires_recording_variable_for_ion_expansion():
    recording = {"type": "CompRecording", "name": "Step.soma"}

    with pytest.raises(KeyError, match="var.*variable"):
        ProtocolConfiguration(
            name="Step",
            stimuli=[],
            recordings_from_config=recording,
            ion_variables=["ina"],
        )
