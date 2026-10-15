"""Tests for evaluation recording helpers."""

from types import SimpleNamespace

import numpy
from bluepyopt import ephys

from bluepyemodel.evaluation.recordings import (
    FixedDtRecordingCustom,
    FixedDtRecordingStimulus,
    LooseDtRecordingCustom,
    LooseDtRecordingStimulus,
    check_recordings,
    get_i_membrane,
    get_loc_currents,
    get_loc_ions,
    get_loc_varlist,
)


class Section:
    def psection(self):
        return {
            "ions": {"na": {"nai": 0.0, "ina": 0.0, "ena": 50.0}},
            "density_mechs": {
                "pas": {"e": -80.0, "i": 0.0},
                "extracellular": {"i_membrane": 0},
            },
        }


def test_recording_location_helpers():
    section = Section()

    assert get_loc_ions(section) == {"nai"}
    assert get_loc_currents(section) == {"ina"}
    assert get_loc_varlist(section) == [
        "e_pas",
        "i_pas",
        "i_membrane_extracellular",
        "v",
    ]
    assert get_i_membrane(section) == ["i_membrane"]


def test_loose_recording_response_converts_current_density():
    recording = LooseDtRecordingCustom("current", variable="ina")
    recording.instantiated = True
    recording.local_ion_list = {"nai"}
    recording.segment_area = 2.0
    recording.tvector = SimpleNamespace(to_python=lambda: [0.0, 1.0])
    recording.varvector = SimpleNamespace(to_python=lambda: [1.0, 2.0])

    response = recording.response

    assert response.response["time"].tolist() == [0.0, 1.0]
    numpy.testing.assert_allclose(response.response["voltage"], [20.0, 40.0])


def test_recording_response_returns_none_before_instantiation():
    recording = LooseDtRecordingCustom("voltage")

    assert recording.response is None


def test_fixed_dt_recording_uses_fixed_interval(monkeypatch):
    calls = []

    class Vector:
        def record(self, reference, *args):
            calls.append(args)

    location = SimpleNamespace(
        instantiate=lambda sim, icell: SimpleNamespace(
            _ref_v="voltage", sec=Section(), area=lambda: 1.0
        )
    )
    sim = SimpleNamespace(
        neuron=SimpleNamespace(h=SimpleNamespace(Vector=Vector, _ref_t="time"))
    )
    recording = FixedDtRecordingCustom("voltage", location=location)

    recording.instantiate(sim=sim, icell=object())

    assert calls == [(0.1,), (0.1,)]


def test_get_i_membrane_returns_empty_when_extracellular_is_missing():
    section = SimpleNamespace(
        psection=lambda: {"ions": {}, "density_mechs": {"pas": {"i": 0.0}}}
    )

    assert get_i_membrane(section) == []


def test_check_recordings_validates_variables_and_reuses_section_cache():
    class CachedSection(Section):
        calls = 0

        def __str__(self):
            return "shared-section"

        def psection(self):
            CachedSection.calls += 1
            return super().psection()

    section = CachedSection()
    valid_location = SimpleNamespace(
        instantiate=lambda sim, icell: SimpleNamespace(sec=section)
    )

    class InvalidLocation:
        def instantiate(self, sim, icell):
            raise ephys.locations.EPhysLocInstantiateException("missing")

    checked = SimpleNamespace(checked=True, location=None, variable="v")
    valid = SimpleNamespace(checked=False, location=valid_location, variable="v")
    cached = SimpleNamespace(checked=False, location=valid_location, variable="nai")
    invalid_variable = SimpleNamespace(
        checked=False, location=valid_location, variable="not_available"
    )
    missing_location = SimpleNamespace(
        checked=False, location=InvalidLocation(), variable="v"
    )

    result = check_recordings(
        [checked, valid, cached, invalid_variable, missing_location], object(), object()
    )

    assert result == [checked, valid, cached]
    assert valid.checked is True
    assert cached.checked is True
    assert invalid_variable.checked is False
    assert CachedSection.calls == 4


def test_loose_custom_recording_instantiates_without_interval():
    calls = []

    class Vector:
        def record(self, reference, *args):
            calls.append((reference, *args))

    location = SimpleNamespace(
        instantiate=lambda sim, icell: SimpleNamespace(
            _ref_v="voltage", sec=Section(), area=lambda: 2.0
        )
    )
    sim = SimpleNamespace(
        neuron=SimpleNamespace(h=SimpleNamespace(Vector=Vector, _ref_t="time"))
    )
    recording = LooseDtRecordingCustom("voltage", location=location)

    recording.instantiate(sim=sim, icell=object())

    assert calls == [("voltage",), ("time",)]
    assert recording.instantiated is True
    assert recording.segment_area == 2.0


def test_stimulus_recordings_use_expected_intervals():
    calls = []

    class Vector:
        def record(self, reference, *args):
            calls.append((reference, *args))

    sim = SimpleNamespace(
        neuron=SimpleNamespace(h=SimpleNamespace(Vector=Vector, _ref_t="time"))
    )
    stimulus = SimpleNamespace(iclamp=SimpleNamespace(_ref_i="current"))

    loose = LooseDtRecordingStimulus("loose", variable="i")
    loose.instantiate(sim=sim, stimulus=stimulus)
    fixed = FixedDtRecordingStimulus("fixed", variable="i")
    fixed.instantiate(sim=sim, stimulus=stimulus)

    assert calls == [("current",), ("time",), ("current", 0.1), ("time", 0.1)]
    assert loose.checked is True
    assert fixed.checked is True
