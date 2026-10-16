"""Deterministic e-code tests covering constructors, properties and waveform generation."""

from types import SimpleNamespace

import numpy
import pytest

from bluepyemodel.ecode.customfromfile import CustomFromFile
from bluepyemodel.ecode.noise import NoiseMixin
from bluepyemodel.ecode.noiseou3 import NoiseOU3
from bluepyemodel.ecode.probampanmda_ems import ProbAMPANMDA_EMS
from bluepyemodel.ecode.sinespec import SineSpec
from bluepyemodel.ecode.square import BPOSquarePulse
from bluepyemodel.ecode.subwhitenoise import SubWhiteNoise
from bluepyemodel.ecode.thresholdaddition import ThresholdAddition
from bluepyemodel.ecode.whitenoise import WhiteNoise

LOCATION = "soma"


def test_bpo_square_pulse_properties_and_generate():
    stimulus = BPOSquarePulse(
        LOCATION,
        amp=0.2,
        delay=100.0,
        duration=200.0,
        totduration=400.0,
        holding_current=-0.05,
    )

    assert stimulus.stim_start == pytest.approx(100.0)
    assert stimulus.stim_end == pytest.approx(300.0)
    assert stimulus.amplitude == pytest.approx(0.2)
    assert "current played at soma" in str(stimulus)

    time, current = stimulus.generate(dt=0.1)

    assert time[0] == pytest.approx(0.0)
    assert len(time) == len(current)
    assert current[0] == pytest.approx(-0.05)
    assert current[1500] == pytest.approx(0.15)
    assert current[-1] == pytest.approx(-0.05)


def test_bpo_square_pulse_defaults_without_holding():
    stimulus = BPOSquarePulse(LOCATION, amp=0.1)

    assert stimulus.stim_start == pytest.approx(250.0)
    assert stimulus.stim_end == pytest.approx(1600.0)

    _, current = stimulus.generate()

    assert current[0] == pytest.approx(0.0)
    assert current[3000] == pytest.approx(0.1)


def test_bpo_square_pulse_destroy_clears_clamps():
    stimulus = BPOSquarePulse(LOCATION, amp=0.1)
    stimulus.iclamp = object()
    stimulus.holding_iclamp = object()

    stimulus.destroy()

    assert stimulus.iclamp is None
    assert stimulus.holding_iclamp is None


def test_threshold_addition_requires_amp():
    # The IDrest base constructor rejects the fully unset case first.
    with pytest.raises(TypeError, match="cannot be both None"):
        ThresholdAddition(LOCATION, amp=None, thresh_perc=None)

    # With only thresh_perc set, ThresholdAddition's own amp guard fires.
    with pytest.raises(TypeError, match="amp cannot be None"):
        ThresholdAddition(LOCATION, amp=None, thresh_perc=120.0)


def test_threshold_addition_amplitude_adds_threshold():
    stimulus = ThresholdAddition(LOCATION, amp=0.05)

    with pytest.raises(ValueError, match="threshold_current should not be None"):
        _ = stimulus.amplitude

    stimulus.threshold_current = 0.2
    assert stimulus.amplitude == pytest.approx(0.25)


def test_sinespec_requires_amp_or_threshold_percentage():
    with pytest.raises(TypeError, match="cannot be both None"):
        SineSpec(LOCATION, amp=None, thresh_perc=None)


def test_sinespec_properties_and_amplitude():
    stimulus = SineSpec(
        LOCATION, amp=0.1, delay=10.0, duration=1000.0, totduration=1100.0
    )

    assert stimulus.stim_start == pytest.approx(10.0)
    assert stimulus.stim_end == pytest.approx(1010.0)
    assert stimulus.amplitude == pytest.approx(0.1)

    stimulus.threshold_current = 0.4
    assert stimulus.amplitude == pytest.approx(0.4 * 0.6)

    absolute = SineSpec(LOCATION, amp=0.1, thresh_perc=None)
    absolute.threshold_current = 0.4
    assert absolute.amplitude == pytest.approx(0.1)


def test_sinespec_generate_adds_oscillation():
    stimulus = SineSpec(
        LOCATION,
        amp=0.1,
        thresh_perc=None,
        delay=10.0,
        duration=100.0,
        totduration=200.0,
        holding_current=-0.02,
    )

    time, current = stimulus.generate(dt=0.1)

    assert len(time) == len(current)
    assert current[0] == pytest.approx(-0.02)
    assert current[-1] == pytest.approx(-0.02)
    assert not numpy.allclose(current[100:1100], -0.02)


def test_sinespec_generate_defaults_holding_to_zero():
    stimulus = SineSpec(
        LOCATION, amp=0.1, thresh_perc=None, duration=100.0, totduration=100.0
    )

    _, current = stimulus.generate()

    assert numpy.isfinite(current).all()


def write_series(path, times, currents):
    numpy.savetxt(path, numpy.transpose(numpy.vstack((times, currents))))
    return str(path)


def test_custom_from_file_returns_stored_series(tmp_path):
    path = write_series(tmp_path / "custom.txt", [0.0, 0.1, 0.2], [0.0, 0.5, 1.0])
    stimulus = CustomFromFile(LOCATION, data_filepath=path)

    assert stimulus.stim_start == pytest.approx(0.0)
    assert stimulus.stim_end == pytest.approx(0.2)
    assert stimulus.total_duration == pytest.approx(0.2)

    time, current = stimulus.generate()

    numpy.testing.assert_allclose(time, [0.0, 0.1, 0.2])
    numpy.testing.assert_allclose(current, [0.0, 0.5, 1.0])


class FakeNoise(NoiseMixin):
    name = "FakeNoise"

    def __init__(self, location, time_series, current_series, mu, holding_current=None):
        self.time_series = numpy.asarray(time_series)
        self.current_series = numpy.asarray(current_series)
        self.mu = mu
        self.holding_current = holding_current
        self.threshold_current = None
        super().__init__(location=location)


def test_noise_mixin_properties_and_generate():
    stimulus = FakeNoise(
        LOCATION, [0.0, 0.1, 0.2], [0.0, 1.0, 2.0], mu=0.4, holding_current=-0.05
    )

    assert stimulus.stim_start == pytest.approx(0.0)
    assert stimulus.stim_end == pytest.approx(0.2)
    assert stimulus.total_duration == pytest.approx(0.2)

    time, current = stimulus.generate(dt=0.1)

    numpy.testing.assert_allclose(time, [0.0, 0.1, 0.2])
    numpy.testing.assert_allclose(current, [0.35, 0.55, 0.75])


def test_noise_mixin_defaults_holding_current_to_zero():
    stimulus = FakeNoise(LOCATION, [0.0, 0.1], [0.0, 1.0], mu=0.2)

    _, current = stimulus.generate()

    numpy.testing.assert_allclose(current, [0.2, 0.3])


def test_noise_mixin_rejects_other_timesteps():
    stimulus = FakeNoise(LOCATION, [0.0, 0.1], [0.0, 1.0], mu=0.2)

    with pytest.raises(ValueError, match="dt has to be 0.1ms"):
        stimulus.generate(dt=0.025)


def test_white_noise_loads_default_resource():
    stimulus = WhiteNoise(LOCATION, mu=0.3, holding_current=-0.01)

    assert stimulus.time_series[0] == pytest.approx(0.0)
    assert len(stimulus.time_series) == len(stimulus.current_series)
    assert stimulus.total_duration > 0.0

    _, current = stimulus.generate()
    assert len(current) == len(stimulus.current_series)


def test_white_noise_loads_custom_file(tmp_path):
    path = write_series(tmp_path / "noise.txt", [0.0, 0.1, 0.2], [1.0, 2.0, 3.0])
    stimulus = WhiteNoise(LOCATION, mu=0.2, data_filepath=path)

    numpy.testing.assert_allclose(stimulus.time_series, [0.0, 0.1, 0.2])
    numpy.testing.assert_allclose(stimulus.current_series, [1.0, 2.0, 3.0])


def test_noise_ou3_loads_default_resource():
    stimulus = NoiseOU3(LOCATION, mu=0.25)

    assert len(stimulus.time_series) == len(stimulus.current_series)
    assert stimulus.stim_end == pytest.approx(stimulus.time_series[-1])


def test_noise_ou3_loads_custom_file(tmp_path):
    path = write_series(tmp_path / "ou3.txt", [0.0, 0.1], [0.5, 1.5])
    stimulus = NoiseOU3(LOCATION, mu=0.1, data_filepath=path)

    numpy.testing.assert_allclose(stimulus.current_series, [0.5, 1.5])


def test_sub_white_noise_loads_default_resource():
    stimulus = SubWhiteNoise(LOCATION, mu=0.15)

    assert len(stimulus.time_series) == len(stimulus.current_series)
    assert stimulus.stim_start == pytest.approx(0.0)


def test_sub_white_noise_loads_custom_file(tmp_path):
    path = write_series(tmp_path / "sub.txt", [0.0, 0.1], [0.25, 0.75])
    stimulus = SubWhiteNoise(LOCATION, mu=0.05, data_filepath=path)

    numpy.testing.assert_allclose(stimulus.time_series, [0.0, 0.1])


def test_prob_ampan_mda_ems_fake_instantiation_and_destroy():
    class Segment:
        x = 0.5
        sec = "section"

    class Location:
        def instantiate(self, sim, icell):
            return Segment()

    class Synapse:
        Use = None

    class NetStim:
        pass

    class Weight:
        def __init__(self):
            self.values = {}

        def __setitem__(self, index, value):
            self.values[index] = value

    class NetCon:
        def __init__(self, *args, **kwargs):
            self.weight = Weight()

    class H:
        def ProbAMPANMDA_EMS(self, *args, **kwargs):
            return Synapse()

        def NetStim(self, **kwargs):
            return NetStim()

        def NetCon(self, *args, **kwargs):
            return NetCon(*args, **kwargs)

    sim = SimpleNamespace(neuron=SimpleNamespace(h=H()))
    stimulus = ProbAMPANMDA_EMS(
        Location(), syn_weight=0.7, syn_delay=12.0, stimfreq=50.0, number=2
    )

    stimulus.instantiate(sim=sim, icell="icell")

    assert stimulus.synapse.Use == 1.0
    assert stimulus.netstim.interval == 20.0
    assert stimulus.netstim.number == 2
    assert stimulus.netstim.start == 12.0
    assert stimulus.netcon.weight.values[0] == 0.7

    stimulus.destroy()
    assert stimulus.synapse is None
    assert stimulus.netstim is None
    assert stimulus.netcon is None
