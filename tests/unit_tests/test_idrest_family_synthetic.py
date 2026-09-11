"""Synthetic coverage for the remaining IDrest-family e-codes."""

import numpy
import pytest

from bluepyemodel.ecode.apwaveform import APWaveform
from bluepyemodel.ecode.firepattern import FirePattern
from bluepyemodel.ecode.hyperdepol import HyperDepol
from bluepyemodel.ecode.idrest import IDrest
from bluepyemodel.ecode.iv import IV


@pytest.mark.parametrize(
    "stimulus_class, defaults",
    [
        (APWaveform, (220.0, 250.0, 50.0, 550.0)),
        (FirePattern, (200.0, 250.0, 3600.0, 4100.0)),
        (IV, (-40.0, 250.0, 3000.0, 3500.0)),
    ],
)
def test_idrest_family_defaults(stimulus_class, defaults):
    stimulus = stimulus_class(location=None, amp=0.2)

    assert (
        stimulus.amp_rel,
        stimulus.delay,
        stimulus.duration,
        stimulus.total_duration,
    ) == defaults
    assert stimulus.amplitude == pytest.approx(0.2)


def test_idrest_family_explicit_values_override_defaults():
    stimulus = APWaveform(
        location=None,
        amp=0.2,
        thresh_perc=150,
        delay=1,
        duration=2,
        totduration=4,
    )
    stimulus.threshold_current = 0.4

    assert stimulus.amplitude == pytest.approx(0.6)
    assert (stimulus.stim_start, stimulus.stim_end) == (1, 3)


def test_idrest_validation_and_generate():
    with pytest.raises(TypeError):
        IDrest(location=None, amp=None, thresh_perc=None)

    stimulus = IDrest(
        location=None,
        amp=None,
        thresh_perc=150,
        holding_current=0.1,
        delay=1,
        duration=3,
        totduration=6,
    )
    stimulus.threshold_current = 0.2

    time, current = stimulus.generate(dt=1)

    numpy.testing.assert_array_equal(time, numpy.arange(6.0))
    numpy.testing.assert_allclose(current, [0.1, 0.4, 0.4, 0.4, 0.1, 0.1])
    assert stimulus.amplitude == pytest.approx(0.3)


def test_hyperdepol_aliases_relative_amplitudes_and_generate():
    stimulus = HyperDepol(
        location=None,
        amp=-0.2,
        amp2=0.4,
        hyper_amp_rel=-100,
        depol_amp_rel=50,
        thresh_perc=-80,
        holding_current=0.1,
        delay=1,
        tmid=3,
        toff=5,
        totduration=7,
    )
    stimulus.threshold_current = 0.2

    time, current = stimulus.generate(dt=1)

    numpy.testing.assert_array_equal(time, numpy.arange(7.0))
    numpy.testing.assert_allclose(current, [0.1, -0.1, -0.1, 0.2, 0.2, 0.1, 0.1])
    assert stimulus.amplitude == pytest.approx(-0.2)
    assert stimulus.depol_amplitude == pytest.approx(0.1)


@pytest.mark.parametrize(
    "kwargs",
    [{"amp": None, "amp2": 0.1}, {"amp": 0.1, "amp2": None}],
)
def test_hyperdepol_requires_both_step_amplitudes(kwargs):
    with pytest.raises(TypeError):
        HyperDepol(location=None, **kwargs)
