"""Deterministic tests for evaluator definition helpers."""

from types import SimpleNamespace

import pytest

from bluepyemodel.evaluation import evaluator as evaluator_module
from bluepyemodel.evaluation.efel_feature_bpem import (
    DendFitFeature,
    DendFitMultiProtocolsFeature,
    eFELFeatureBPEM,
)
from bluepyemodel.evaluation.protocol_configuration import ProtocolConfiguration
from bluepyemodel.evaluation.recordings import (
    FixedDtRecordingCustom,
    FixedDtRecordingStimulus,
    LooseDtRecordingCustom,
    LooseDtRecordingStimulus,
)


def test_define_location_defaults_to_soma():
    assert evaluator_module.define_location(None) is evaluator_module.soma_loc
    assert evaluator_module.define_location("soma") is evaluator_module.soma_loc


def test_define_location_comp_recording_soma_and_ais():
    soma = {"type": "CompRecording", "location": "soma", "name": "prot.soma.v"}
    ais = {"type": "CompRecording", "location": "ais", "name": "prot.ais.v"}

    assert evaluator_module.define_location(soma) is evaluator_module.soma_loc
    assert evaluator_module.define_location(ais) is evaluator_module.ais_loc


def test_define_location_comp_recording_rejects_other_locations():
    definition = {
        "type": "CompRecording",
        "location": "apical",
        "name": "prot.apical.v",
    }

    with pytest.raises(ValueError, match="Only soma and ais are implemented"):
        evaluator_module.define_location(definition)


def test_define_location_distance_and_seclist_variants():
    somadistance = evaluator_module.define_location(
        {
            "type": "somadistance",
            "name": "dend100",
            "somadistance": 100,
            "seclist_name": "apical",
        }
    )
    assert somadistance.soma_distance == 100

    apic = evaluator_module.define_location(
        {
            "type": "somadistanceapic",
            "name": "apic100",
            "somadistance": 100,
            "seclist_name": "apical",
        }
    )
    assert apic.direction == "radial"

    seclist = evaluator_module.define_location(
        {
            "type": "nrnseclistcomp",
            "name": "axon0",
            "comp_x": 0.5,
            "sec_index": 0,
            "seclist_name": "axonal",
        }
    )
    assert seclist.sec_index == 0


def test_define_location_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown location type"):
        evaluator_module.define_location({"type": "mystery", "name": "x"})


def test_define_recording_stimulus_variants():
    fixed = evaluator_module.define_recording(
        {"type": "FixedDtRecordingStimulus", "name": "prot.soma.i", "variable": "i"}
    )
    loose = evaluator_module.define_recording(
        {"type": "LooseDtRecordingStimulus", "name": "prot.soma.i", "variable": "i"}
    )
    promoted = evaluator_module.define_recording(
        {"type": "LooseDtRecordingStimulus", "name": "prot.soma.i", "variable": "i"},
        use_fixed_dt_recordings=True,
    )

    assert isinstance(fixed, FixedDtRecordingStimulus)
    assert isinstance(loose, LooseDtRecordingStimulus)
    assert isinstance(promoted, FixedDtRecordingStimulus)


def test_define_recording_location_variants_and_var_alias():
    loose = evaluator_module.define_recording(
        {"type": "CompRecording", "name": "prot.soma.v", "location": "soma", "var": "v"}
    )
    fixed = evaluator_module.define_recording(
        {
            "type": "CompRecording",
            "name": "prot.soma.v",
            "location": "soma",
            "variable": "v",
        },
        use_fixed_dt_recordings=True,
    )

    assert isinstance(loose, LooseDtRecordingCustom)
    assert loose.variable == "v"
    assert isinstance(fixed, FixedDtRecordingCustom)


def make_protocol_configuration(
    name="IDrest_100", protocol_type="Protocol", stimuli=None, stochasticity=False
):
    return ProtocolConfiguration(
        name=name,
        stimuli=stimuli
        or [
            {
                "delay": 250.0,
                "amp": 0.2,
                "thresh_perc": None,
                "duration": 1350.0,
                "totduration": 1850.0,
                "holding_current": -0.05,
            }
        ],
        recordings_from_config=[
            {
                "type": "CompRecording",
                "name": f"{name}.soma.v",
                "location": "soma",
                "variable": "v",
            }
        ],
        protocol_type=protocol_type,
        stochasticity=stochasticity,
    )


def test_define_protocol_builds_simple_protocol():
    protocol = evaluator_module.define_protocol(make_protocol_configuration())

    assert protocol.name == "IDrest_100"
    assert protocol.cvode_active is True
    assert len(protocol.recordings) == 1


def test_define_protocol_disables_cvode_for_fixed_dt():
    protocol = evaluator_module.define_protocol(make_protocol_configuration(), dt=0.025)

    assert protocol.cvode_active is False


def test_define_protocol_rejects_multiple_stimuli():
    configuration = make_protocol_configuration()
    configuration.stimuli = configuration.stimuli * 2

    with pytest.raises(ValueError, match="single stimulus"):
        evaluator_module.define_protocol(configuration)


def test_define_protocol_rejects_unknown_ecode():
    configuration = make_protocol_configuration(name="NotAnEcode_100")

    with pytest.raises(KeyError, match="no eCode linked"):
        evaluator_module.define_protocol(configuration)


def test_define_protocol_rejects_unknown_protocol_type():
    configuration = make_protocol_configuration(protocol_type="Mystery")

    with pytest.raises(ValueError, match="not found"):
        evaluator_module.define_protocol(configuration)


def test_define_protocol_threshold_based_dependencies():
    configuration = make_protocol_configuration(protocol_type="ThresholdBasedProtocol")

    protocol = evaluator_module.define_protocol(configuration)

    assert protocol.dependencies["stimulus.holding_current"] == [
        "SearchHoldingCurrent",
        "bpo_holding_current",
    ]


def test_define_protocol_no_holding_variant_keys():
    configuration = make_protocol_configuration(
        protocol_type="ThresholdBasedNoHoldingProtocol"
    )

    protocol = evaluator_module.define_protocol(configuration)

    assert protocol.dependencies["stimulus.holding_current"] == [
        "SearchHoldingCurrent_noholding",
        "bpo_holding_current_noholding",
    ]


def make_feature_config(
    name="IDrest_100.soma.v.mean_frequency",
    efel_feature_name="mean_frequency",
    recording_names=None,
    efel_settings=None,
):
    return SimpleNamespace(
        name=name,
        efel_feature_name=efel_feature_name,
        recording_name_for_instantiation=recording_names or {"": "IDrest_100.soma.v"},
        mean=6.0,
        std=1.0,
        weight=1.0,
        efel_settings=efel_settings if efel_settings is not None else {},
    )


def make_fake_protocol(name="IDrest_100", total_duration=1850.0):
    return SimpleNamespace(
        name=name,
        amplitude=0.2,
        stim_start=lambda: 250.0,
        stim_end=lambda: 1600.0,
        total_duration=total_duration,
        stimuli=[SimpleNamespace()],
    )


def test_define_efeature_uses_protocol_bounds():
    efeature = evaluator_module.define_efeature(
        make_feature_config(), make_fake_protocol()
    )

    assert isinstance(efeature, eFELFeatureBPEM)
    assert efeature.stim_start == 250.0
    assert efeature.stim_end == 1600.0
    assert efeature.stimulus_current == pytest.approx(0.2)


def test_define_efeature_prefers_explicit_efel_settings():
    config = make_feature_config(efel_settings={"stim_start": 10.0, "stim_end": 20.0})

    efeature = evaluator_module.define_efeature(config, make_fake_protocol())

    assert efeature.stim_start == 10.0
    assert efeature.stim_end == 20.0


def test_define_efeature_uses_total_duration_for_bAP():
    efeature = evaluator_module.define_efeature(
        make_feature_config(),
        make_fake_protocol(name="bAP_1000", total_duration=1000.0),
    )

    assert efeature.stim_end == 1000.0


def test_define_efeature_splits_settings_by_type():
    config = make_feature_config(
        efel_settings={
            "Threshold": -30.0,
            "interp_step": 0.025,
            "strict_stiminterval": 1,
            "mode": "mean",
        }
    )

    efeature = evaluator_module.define_efeature(config, make_fake_protocol())

    assert efeature.threshold == pytest.approx(-30.0)
    assert efeature.interp_step == pytest.approx(0.025)
    assert efeature.int_settings["strict_stiminterval"] == 1
    assert efeature.string_settings["mode"] == "mean"


def test_define_efeature_builds_dend_fit_feature():
    config = make_feature_config(
        name="apical_dendrite_backpropagation_fit_decay",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names={"": "bAP_1000.soma.v", "100": "bAP_1000.apical100.v"},
    )

    efeature = evaluator_module.define_efeature(config, make_fake_protocol())

    assert isinstance(efeature, DendFitFeature)
    assert efeature.decay is True
    assert efeature.linear is False


def test_define_efeature_builds_multiprotocol_feature():
    config = make_feature_config(
        name="apical_fit_linear",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names={"": "IDrestapical[050,100]_100.apical.v"},
    )

    efeature = evaluator_module.define_efeature(config, make_fake_protocol())

    assert isinstance(efeature, DendFitMultiProtocolsFeature)
    assert efeature.linear is True


def test_define_efeature_handles_multiple_decay_time_constant():
    config = make_feature_config(
        efel_feature_name="multiple_decay_time_constant_after_stim",
        efel_settings={},
    )
    protocol = make_fake_protocol()

    efeature = evaluator_module.define_efeature(config, protocol)

    assert efeature.double_settings["multi_stim_start"] == [250.0]
    assert efeature.double_settings["multi_stim_end"] == [1600.0]


def test_define_efeature_uses_stimulus_multi_bounds():
    config = make_feature_config(
        efel_feature_name="multiple_decay_time_constant_after_stim",
        efel_settings={},
    )
    protocol = make_fake_protocol()
    protocol.stimuli = [
        SimpleNamespace(
            multi_stim_start=lambda: [100.0, 200.0],
            multi_stim_end=lambda: [150.0, 250.0],
        )
    ]

    efeature = evaluator_module.define_efeature(config, protocol)

    assert efeature.double_settings["multi_stim_start"] == [100.0, 200.0]
    assert efeature.double_settings["multi_stim_end"] == [150.0, 250.0]


def test_define_protocols_skips_validation_when_excluded():
    optimisation = make_protocol_configuration(name="IDrest_100")
    validation = make_protocol_configuration(name="APWaveform_200")
    validation.validation = True
    configuration = SimpleNamespace(protocols=[optimisation, validation])

    protocols = evaluator_module.define_protocols(
        configuration, False, False, False, None
    )

    assert set(protocols) == {"IDrest_100"}

    all_protocols = evaluator_module.define_protocols(
        configuration, True, False, False, None
    )
    assert set(all_protocols) == {"IDrest_100", "APWaveform_200"}


def test_define_efeatures_requires_existing_protocol():
    feature = SimpleNamespace(
        protocol_name="Absent_100",
        name="Absent_100.soma.v.mean_frequency",
        efel_feature_name="mean_frequency",
        recording_name_for_instantiation={"": "Absent_100.soma.v"},
        mean=1.0,
        std=1.0,
        weight=1.0,
        efel_settings={},
    )
    configuration = SimpleNamespace(efeatures=[feature], validation_protocols=[])

    with pytest.raises(ValueError, match="Could not find protocol named Absent_100"):
        evaluator_module.define_efeatures(configuration, True, {}, {})


def test_define_efeatures_skips_validation_features():
    feature = SimpleNamespace(
        protocol_name="APWaveform_200",
        name="APWaveform_200.soma.v.AP_amplitude",
        efel_feature_name="AP_amplitude",
        recording_name_for_instantiation={"": "APWaveform_200.soma.v"},
        mean=1.0,
        std=1.0,
        weight=1.0,
        efel_settings={},
    )
    configuration = SimpleNamespace(
        efeatures=[feature], validation_protocols=["APWaveform_200"]
    )

    assert evaluator_module.define_efeatures(configuration, False, {}, {}) == []


def test_define_efeatures_allows_pre_protocol_without_protocol():
    feature = SimpleNamespace(
        protocol_name="RMPProtocol",
        name="RMPProtocol.soma.v.steady_state_voltage_stimend",
        efel_feature_name="steady_state_voltage_stimend",
        recording_name_for_instantiation={"": "RMPProtocol.soma.v"},
        mean=-75.0,
        std=2.0,
        weight=1.0,
        efel_settings={},
    )
    configuration = SimpleNamespace(efeatures=[feature], validation_protocols=[])

    efeatures = evaluator_module.define_efeatures(configuration, True, {}, {})

    assert len(efeatures) == 1
    assert efeatures[0].stim_start is None


def test_define_fitness_calculator_wraps_features():
    features = [
        eFELFeatureBPEM(
            "a", efel_feature_name="mean_frequency", exp_mean=1.0, exp_std=1.0
        ),
        eFELFeatureBPEM(
            "b", efel_feature_name="voltage_base", exp_mean=1.0, exp_std=1.0
        ),
    ]

    calculator = evaluator_module.define_fitness_calculator(features)

    assert len(calculator.objectives) == 2


def make_preprotocol_feature(protocol, feature_name, exp_mean=1.0):
    return eFELFeatureBPEM(
        f"{protocol}.{feature_name}",
        efel_feature_name=feature_name,
        recording_names={"": f"{protocol}.soma.v"},
        exp_mean=exp_mean,
        exp_std=1.0,
    )


def test_define_preprotocol_helpers_construct_and_configure_features():
    rmp_feature = make_preprotocol_feature(
        "RMPProtocol", "steady_state_voltage_stimend", exp_mean=-70.0
    )
    rmp_other = make_preprotocol_feature("RMPProtocol", "voltage_base", exp_mean=-71.0)
    rin_feature = make_preprotocol_feature(
        "RinProtocol", "ohmic_input_resistance_vb_ssse", exp_mean=100.0
    )
    holding_feature = make_preprotocol_feature(
        "SearchHoldingCurrent", "steady_state_voltage_stimend", exp_mean=-70.0
    )
    efeatures = [rmp_feature, rmp_other, rin_feature, holding_feature]

    rmp = evaluator_module.define_RMP_protocol(efeatures, recording_name="custom.v")
    rin = evaluator_module.define_Rin_protocol(efeatures, recording_name="custom.v")
    holding = evaluator_module.define_holding_protocol(
        efeatures, recording_name="custom.v"
    )
    threshold = evaluator_module.define_threshold_protocol(efeatures)

    assert rmp.target_voltage.recording_names == {"": "RMPProtocol.custom.v"}
    assert rmp_other.stim_start == 0.0
    assert rmp_other.stim_end == rmp.stimulus_duration
    assert rin.target_rin.recording_names == {"": "RinProtocol.custom.v"}
    assert holding.target_voltage.recording_names == {
        "": "SearchHoldingCurrent.custom.v"
    }
    assert threshold.target_threshold is None


def test_define_preprotocol_helpers_require_target_features():
    with pytest.raises(ValueError, match="steady_state_voltage_stimend"):
        evaluator_module.define_RMP_protocol([])
    with pytest.raises(ValueError, match="ohmic_input_resistance"):
        evaluator_module.define_Rin_protocol([])
    with pytest.raises(ValueError, match="bpo_holding_current"):
        evaluator_module.define_holding_protocol([])


def test_define_preprotocols_supports_no_holding_branch():
    efeatures = [
        make_preprotocol_feature("RMPProtocol", "steady_state_voltage_stimend", -70.0),
        make_preprotocol_feature(
            "RinProtocol", "ohmic_input_resistance_vb_ssse", 100.0
        ),
        make_preprotocol_feature(
            "SearchHoldingCurrent", "steady_state_voltage_stimend", -70.0
        ),
    ]
    configuration = SimpleNamespace(
        rmp_duration=500.0,
        search_holding_duration=500.0,
        rin_step_amp=-0.02,
        rin_step_delay=500.0,
        rin_step_duration=500.0,
        rin_totduration=1000.0,
        search_threshold_step_delay=500.0,
        search_threshold_step_duration=2000.0,
        search_threshold_totduration=3000.0,
    )

    protocols = evaluator_module.define_preprotocols(
        efeatures, evaluator_module.soma_loc, configuration, no_holding=True
    )

    assert isinstance(
        protocols["SearchHoldingCurrent"], evaluator_module.NoHoldingCurrent
    )
    assert protocols["SearchThresholdCurrent"].no_spikes is False


def test_get_simulator_selects_stochastic_fixed_dt_and_mechanism_parent(
    monkeypatch, tmp_path
):
    calls = []

    class FakeSimulator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(evaluator_module, "NrnSimulator", FakeSimulator)
    cell_model = SimpleNamespace(mechanisms=[SimpleNamespace(deterministic=False)])

    result = evaluator_module.get_simulator(
        stochasticity=True,
        cell_model=cell_model,
        dt=None,
        mechanisms_directory=tmp_path / "mechanisms",
    )

    assert isinstance(result, FakeSimulator)
    assert calls == [
        {
            "dt": 0.025,
            "cvode_active": False,
            "mechanisms_directory": str(tmp_path),
        }
    ]


def test_get_simulator_handles_deterministic_and_variable_dt(monkeypatch, caplog):
    calls = []

    class FakeSimulator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(evaluator_module, "NrnSimulator", FakeSimulator)
    cell_model = SimpleNamespace(mechanisms=[SimpleNamespace(deterministic=True)])

    evaluator_module.get_simulator(True, cell_model, mechanisms_directory=None)
    evaluator_module.get_simulator(
        False, cell_model, dt=0.05, mechanisms_directory=None
    )

    assert calls == [
        {"mechanisms_directory": None, "cvode_minstep": 0.0},
        {"dt": 0.05, "cvode_active": False, "mechanisms_directory": None},
    ]
    assert "no mechanisms are stochastic" in caplog.text


def test_add_recordings_to_evaluator_adds_fixed_and_loose_variables():
    base = SimpleNamespace(name="IDrest_100.soma.v", location=evaluator_module.soma_loc)
    protocol = SimpleNamespace(name="IDrest_100", recordings=[base])
    preprotocol = SimpleNamespace(name="RMPProtocol", recordings=[base])
    evaluator = SimpleNamespace(
        fitness_protocols={
            "main_protocol": SimpleNamespace(
                protocols={"IDrest_100": protocol, "RMPProtocol": preprotocol}
            )
        }
    )

    evaluator_module.add_recordings_to_evaluator(evaluator, ["ina", "cai"], True)

    assert [rec.name for rec in protocol.recordings] == [
        "IDrest_100.soma.v",
        "IDrest_100.soma.ina",
        "IDrest_100.soma.cai",
    ]
    assert len(preprotocol.recordings) == 1
    assert evaluator_module.FixedDtRecordingCustom in {
        type(rec) for rec in protocol.recordings[1:]
    }
