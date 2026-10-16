"""Physical current replay never reapplies holding or rheobase scaling."""

import os
from types import SimpleNamespace

import numpy
import pytest

from bluepyemodel.ecode import eCodes
from bluepyemodel.ecode.customfromfile import CustomFromFile
from bluepyemodel.ecode.recordedcurrent import RecordedCurrent
from bluepyemodel.evaluation.evaluator import define_protocol
from bluepyemodel.evaluation.fitness_calculator_configuration import FitnessCalculatorConfiguration
from bluepyemodel.evaluation.protocol_configuration import ProtocolConfiguration
from bluepyemodel.evaluation.protocols import ThresholdBasedProtocol


def payload():
    return {
        "waveform": [0.2, 0.25, 0.18, 0.22],
        "time_series": [0.0, 0.1, 0.2, 0.3],
        "waveform_includes_holding": True,
        "waveform_unit": "nA",
        "time_unit": "ms",
        "delay": 0.0,
        "duration": 0.3,
        "totduration": 0.3,
    }


def test_replay_does_not_rescale_or_add_holding():
    data = payload()
    stimulus = RecordedCurrent(
        location=None, **data, holding_current=10.0, amp=999, thresh_perc=150
    )
    stimulus.threshold_current = 12.0
    time, current = stimulus.generate(dt=0.025)
    assert current.tobytes() == numpy.asarray(data["waveform"]).tobytes()
    assert time.tobytes() == numpy.asarray(data["time_series"]).tobytes()
    assert stimulus.total_duration == 0.3 and stimulus.stim_start == 0 and stimulus.stim_end == 0.3
    assert stimulus.amplitude == 999
    current[:] = 0
    assert stimulus.generate()[1].tobytes() == numpy.asarray(data["waveform"]).tobytes()


def test_uniform_clock_and_feature_window():
    stimulus = RecordedCurrent(
        location=None, waveform=[0.1, 0.2, 0.3, 0.4], dt=0.1, delay=0.1, duration=0.1
    )
    assert eCodes["recordedcurrent"] is RecordedCurrent
    numpy.testing.assert_array_equal(stimulus.generate()[0], numpy.arange(4) * 0.1)
    assert stimulus.stim_start == 0.1 and stimulus.stim_end == 0.2
    assert stimulus.amplitude == pytest.approx(numpy.std([0.1, 0.2, 0.3, 0.4]))


@pytest.mark.parametrize("name", ["WhiteNoise", "SubWhiteNoise", "NoiseOU3", "RecordedCurrent"])
def test_evaluator_routes_physical_payload_without_threshold_dependencies(name):
    cfg = ProtocolConfiguration(
        name=name,
        stimuli=[payload()],
        recordings_from_config=[],
        protocol_type="ThresholdBasedProtocol",
    )
    protocol = define_protocol(cfg)
    assert isinstance(protocol.stimulus, RecordedCurrent)
    assert not isinstance(protocol, ThresholdBasedProtocol)
    assert (
        protocol.stimulus.generate()[1].tobytes() == numpy.asarray(payload()["waveform"]).tobytes()
    )


def test_extracted_physical_protocol_is_persisted_as_absolute():
    config = FitnessCalculatorConfiguration(
        name_rmp_protocol="RMPProtocol", name_rin_protocol="RinProtocol"
    )
    config._add_bluepyefe_protocol("WhiteNoise", {"step": payload(), "holding": {"amp": 2.0}})
    assert config.protocols[0].protocol_type == "Protocol"
    stimulus = define_protocol(config.protocols[0]).stimulus
    assert stimulus.generate()[1].tobytes() == numpy.asarray(payload()["waveform"]).tobytes()


@pytest.mark.parametrize(
    "updates",
    [
        {"waveform": []},
        {"waveform": [0.1]},
        {"waveform": [0.1, numpy.nan, 0.2, 0.3]},
        {"waveform": [[0.1, 0.2], [0.3, 0.4]]},
        {"time_series": [0.0, 0.1]},
        {"time_series": [0.0, 0.1, 0.1, 0.3]},
        {"time_series": [-0.1, 0.0, 0.1, 0.2]},
        {"time_series": [0.0, 0.1, numpy.inf, 0.3]},
        {"totduration": 0.4},
        {"delay": -1},
        {"duration": 0.4},
        {"duration": 0},
        {"delay": numpy.nan},
        {"waveform_includes_holding": False},
        {"waveform_unit": "A"},
        {"time_unit": "s"},
        {"data_filepath": "ambiguous.txt"},
        {"amp": numpy.nan},
    ],
)
def test_invalid_or_ambiguous_replay_rejected(updates):
    with pytest.raises(ValueError):
        RecordedCurrent(location=None, **{**payload(), **updates})


@pytest.mark.parametrize("dt", [0, -0.1, numpy.nan])
def test_invalid_uniform_clock_rejected(dt):
    with pytest.raises(ValueError):
        RecordedCurrent(location=None, waveform=[0.1, 0.2], dt=dt)


def test_neuron_current_and_voltage_match_text_replay(tmp_path):
    from neuron import h

    time = numpy.arange(201) * 0.1
    current = 0.1 + 0.02 * numpy.sin(time)
    text = tmp_path / "current.txt"
    numpy.savetxt(text, numpy.column_stack((time, current)))
    results = []
    h.load_file("stdrun.hoc")
    for kind in ("recorded", "text"):
        section = h.Section()
        section.L = section.diam = 20
        section.insert("pas")
        location = SimpleNamespace(instantiate=lambda **_: SimpleNamespace(sec=section, x=0.5))
        if kind == "recorded":
            stimulus = RecordedCurrent(location=location, waveform=current, time_series=time)
        else:
            stimulus = CustomFromFile(location=location, data_filepath=text)
        stimulus.instantiate(sim=SimpleNamespace(neuron=SimpleNamespace(h=h)))
        h.cvode_active(0)
        h.dt = 0.025
        voltage = h.Vector().record(section(0.5)._ref_v)
        injected = h.Vector().record(stimulus.iclamp._ref_i)
        h.finitialize(-65)
        h.continuerun(float(time[-1]))
        results.append((numpy.asarray(voltage).copy(), numpy.asarray(injected).copy()))
        stimulus.destroy()
        h.delete_section(sec=section)
    for left, right in zip(results[0], results[1]):
        assert left.tobytes() == right.tobytes()


@pytest.mark.skipif(not os.environ.get("BLUEPY_NOISE_NWB"), reason="External NWB not configured")
def test_real_nwb_extraction_to_evaluator_replay():
    from bluepyefe.cell import Cell

    for name, expected in (("SubWhiteNoise", 4), ("WhiteNoise", 1), ("NoiseOU3", 1)):
        cell = Cell("NWB")
        cell.read_recordings(
            [{"filepath": os.environ["BLUEPY_NOISE_NWB"], "protocol_name": name}],
            name,
            efel_settings={"Threshold": -20.0, "interp_step": 0.025},
        )
        assert len(cell.recordings) == expected
        for recording in cell.recordings:
            metadata = recording.get_stimulus_parameters()
            metadata["holding_current"] = 100.0
            cfg = ProtocolConfiguration(
                name=name,
                stimuli=[metadata],
                recordings_from_config=[],
                protocol_type="ThresholdBasedProtocol",
            )
            replay = define_protocol(cfg).stimulus
            assert replay.generate()[0].tobytes() == recording.generate()[0].tobytes()
            assert replay.generate()[1].tobytes() == recording.generate()[1].tobytes()
