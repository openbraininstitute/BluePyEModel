"""Synthetic coverage for DendriticStep and Ramp e-codes."""

import numpy
import pytest
from bluepyopt.ephys.locations import (
    NrnSomaDistanceCompLocation,
    NrnTrunkSomaDistanceCompLocation,
)

from bluepyemodel.ecode.dendrite import DendriticStep
from bluepyemodel.ecode.idrest import IDrest
from bluepyemodel.ecode.ramp import Ramp


def test_dendritic_step_builds_apical_trunk_location():
    stimulus = DendriticStep(
        location=None,
        direction="apical_trunk",
        somadistance=120,
        sec_index=2,
        amp=0.1,
    )

    assert isinstance(stimulus.location, NrnTrunkSomaDistanceCompLocation)
    assert stimulus.location.soma_distance == 120
    assert stimulus.location.sec_index == 2


def test_dendritic_step_builds_random_location():
    stimulus = DendriticStep(
        location=None,
        direction="random",
        somadistance=80,
        seclist_name="basal",
        amp=0.1,
    )

    assert isinstance(stimulus.location, NrnSomaDistanceCompLocation)
    assert stimulus.location.soma_distance == 80
    assert stimulus.location.seclist_name == "basal"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"direction": "apical_trunk", "seclist_name": "basal"}, "must be apical"),
        ({"direction": "unknown"}, "not understood"),
    ],
)
def test_dendritic_step_rejects_invalid_direction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DendriticStep(location=None, somadistance=50, amp=0.1, **kwargs)


def test_dendritic_step_forces_zero_holding_current(monkeypatch):
    observed = {}

    def fake_instantiate(self, sim=None, icell=None):
        observed["holding_current"] = self.holding_current

    monkeypatch.setattr(IDrest, "instantiate", fake_instantiate)
    stimulus = DendriticStep(
        location=None, somadistance=50, amp=0.1, direction="random"
    )
    stimulus.holding_current = 0.7

    stimulus.instantiate()

    assert observed["holding_current"] == 0


def test_ramp_relative_amplitude_and_generate():
    stimulus = Ramp(
        location=None,
        amp=None,
        thresh_perc=150,
        holding_current=0.5,
        delay=1,
        duration=4,
        totduration=7,
    )
    stimulus.threshold_current = 2.0

    time, current = stimulus.generate(dt=1)

    numpy.testing.assert_array_equal(time, numpy.arange(7.0))
    numpy.testing.assert_allclose(current, [0.5, 0.5, 1.25, 2.0, 2.75, 0.5, 0.5])
    assert stimulus.stim_start == 1
    assert stimulus.stim_end == 5
    assert stimulus.amplitude == pytest.approx(3.0)


def test_ramp_requires_amplitude_and_supports_default_holding():
    with pytest.raises(TypeError):
        Ramp(location=None, amp=None, thresh_perc=None)

    stimulus = Ramp(
        location=None, amp=0.2, holding_current=None, delay=1, duration=2, totduration=4
    )
    _, current = stimulus.generate(dt=1)

    numpy.testing.assert_allclose(current, [0.0, 0.0, 0.1, 0.0])
