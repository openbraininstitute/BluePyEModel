"""Synthetic coverage for deterministic non-NEURON e-code paths."""

import numpy
import pytest

from bluepyemodel.ecode.comb import Comb
from bluepyemodel.ecode.dehyperpol import DeHyperpol
from bluepyemodel.ecode.random_square_inputs import MultipleRandomStepInputs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amp": 0.1},
        {"amp2": -0.1},
    ],
)
def test_dehyperpol_requires_both_step_amplitudes(kwargs):
    with pytest.raises(TypeError):
        DeHyperpol(location=None, **kwargs)


def test_dehyperpol_threshold_alias_and_relative_amplitudes():
    stimulus = DeHyperpol(
        location=None,
        amp=0.2,
        amp2=-0.4,
        amp_rel=50,
        amp2_rel=-150,
        thresh_perc=-100,
    )

    assert stimulus.amplitude == pytest.approx(-0.4)
    assert stimulus.depol_amplitude == pytest.approx(0.2)

    stimulus.threshold_current = 0.2
    assert stimulus.amplitude == pytest.approx(-0.3)
    assert stimulus.depol_amplitude == pytest.approx(0.1)


def test_dehyperpol_relative_aliases_and_generate():
    stimulus = DeHyperpol(
        location=None,
        amp_rel=50,
        thresh_perc=-100,
        holding_current=0.1,
        delay=1,
        tmid=3,
        toff=5,
        totduration=7,
    )
    stimulus.threshold_current = 0.2

    time, current = stimulus.generate(dt=1)

    numpy.testing.assert_array_equal(time, numpy.arange(7.0))
    numpy.testing.assert_allclose(current, [0.1, 0.2, 0.2, -0.1, -0.1, 0.1, 0.1])
    assert stimulus.stim_start == 1
    assert stimulus.stim_end == 5


def test_comb_generate_with_inter_delay():
    stimulus = Comb(
        location=None,
        delay=1,
        inter_delay=3,
        n_steps=3,
        duration=1,
        amp=2,
        totduration=10,
    )

    time, current = stimulus.generate(dt=1)

    numpy.testing.assert_array_equal(time, numpy.arange(10.0))
    numpy.testing.assert_array_equal(current, [0, 2, 0, 0, 2, 0, 0, 2, 0, 0])
    assert stimulus.stim_start == 1
    assert stimulus.stim_end == 4
    assert stimulus.amplitude == 2


def test_comb_rejects_steps_beyond_total_duration():
    with pytest.raises(ValueError):
        Comb(
            location=None,
            delay=1,
            inter_delay=3,
            n_steps=3,
            duration=2,
            totduration=6,
        )


def test_random_step_inputs_are_seeded_and_use_relative_amplitude():
    numpy.random.seed(7)
    first = MultipleRandomStepInputs(
        location=None,
        amp=None,
        thresh_perc=150,
        delay=2,
        duration=4,
        n_inputs=5,
        sections=["dend"],
    )
    numpy.random.seed(7)
    second = MultipleRandomStepInputs(
        location=None,
        amp=None,
        thresh_perc=150,
        delay=2,
        duration=4,
        n_inputs=5,
        sections=["dend"],
    )

    assert first.inputs_start == second.inputs_start
    assert len(first.inputs_start) == 5
    assert all(2 <= start < 6 for start in first.inputs_start)
    assert first.amplitude is None

    first.threshold_current = 0.2
    assert first.amplitude == pytest.approx(0.3)
    assert first.stim_start == 2
    assert first.stim_end == 6


def test_random_step_inputs_zero_inputs_and_inherited_generate():
    stimulus = MultipleRandomStepInputs(
        location=None,
        amp=0.1,
        n_inputs=0,
        holding_current=-0.02,
    )

    time, current = stimulus.generate()

    assert stimulus.inputs_start == []
    assert time == []
    assert current == []
