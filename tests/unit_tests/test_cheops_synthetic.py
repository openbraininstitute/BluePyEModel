"""Synthetic coverage for the deterministic Cheops e-codes."""

import numpy
import pytest

from bluepyemodel.ecode.negcheops import NegCheops
from bluepyemodel.ecode.poscheops import PosCheops


@pytest.mark.parametrize(
    ("stimulus_class", "kwargs", "error"),
    [
        (PosCheops, {"amp": None, "thresh_perc": None}, TypeError),
        (NegCheops, {"amp": None, "thresh_perc": None}, ValueError),
        (PosCheops, {"amp": 0.1, "holding_current": 0.2}, ValueError),
        (NegCheops, {"amp": -0.1, "holding_current": -0.2}, ValueError),
    ],
)
def test_cheops_constructor_validation(stimulus_class, kwargs, error):
    with pytest.raises(error):
        stimulus_class(location=None, **kwargs)


def test_poscheops_properties_and_generate():
    stimulus = PosCheops(
        location=None,
        amp=0.5,
        thresh_perc=200,
        holding_current=0.1,
        delay=1.0,
        t1=3.0,
        t2=4.0,
        t3=6.0,
        t4=7.0,
        toff=9.0,
        totduration=10.0,
    )
    stimulus.threshold_current = 0.2

    time, current = stimulus.generate(dt=1.0)

    assert stimulus.stim_start == 1.0
    assert stimulus.stim_end == 9.0
    assert stimulus.amplitude == 0.4
    numpy.testing.assert_allclose(time, numpy.arange(10.0))
    assert current[0] == pytest.approx(0.1)
    assert current.max() == pytest.approx(0.6)
    assert current[-1] == pytest.approx(0.1)


def test_negcheops_properties_and_generate():
    stimulus = NegCheops(
        location=None,
        amp=-0.5,
        thresh_perc=-100,
        holding_current=0.0,
        delay=1.0,
        t1=3.0,
        t2=4.0,
        t3=6.0,
        t4=7.0,
        toff=9.0,
        totduration=10.0,
    )
    stimulus.threshold_current = 0.2

    time, current = stimulus.generate(dt=1.0)

    assert stimulus.stim_start == 1.0
    assert stimulus.stim_end == 9.0
    assert stimulus.amplitude == pytest.approx(-0.2)
    numpy.testing.assert_allclose(time, numpy.arange(10.0))
    assert current[0] == pytest.approx(0.0)
    assert current.min() == pytest.approx(-0.5)
    assert current[-1] == pytest.approx(0.0)


def test_cheops_constructor_derives_ramp_times():
    pos = PosCheops(
        location=None,
        amp=0.2,
        holding_current=0.0,
        delay=2.0,
        ramp1_duration=1.0,
        ramp2_duration=2.0,
        ramp3_duration=3.0,
        inter_delay=4.0,
    )
    neg = NegCheops(
        location=None,
        amp=-0.2,
        delay=2.0,
        ramp1_duration=1.0,
        ramp2_duration=2.0,
        ramp3_duration=3.0,
        inter_delay=4.0,
    )

    assert (pos.t1, pos.t2, pos.t3, pos.t4, pos.toff, pos.total_duration) == (
        4.0,
        8.0,
        12.0,
        16.0,
        22.0,
        22.0 + 250.0,
    )
    assert (neg.t1, neg.t2, neg.t3, neg.t4, neg.toff) == (4.0, 8.0, 12.0, 16.0, 22.0)
