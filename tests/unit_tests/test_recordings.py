"""Tests for evaluation recording helpers."""

from types import SimpleNamespace

import numpy

from bluepyemodel.evaluation.recordings import (
    FixedDtRecordingCustom,
    LooseDtRecordingCustom,
    get_i_membrane,
    get_loc_currents,
    get_loc_ions,
    get_loc_varlist,
)


class Section:
    def psection(self):
        return {
            "ions": {"na": {"nai": 0.0, "ina": 0.0, "ena": 50.0}},
            "density_mechs": {"pas": {"e": -80.0, "i": 0.0}, "extracellular": {"i_membrane": 0}},
        }


def test_recording_location_helpers():
    section = Section()

    assert get_loc_ions(section) == {"nai"}
    assert get_loc_currents(section) == {"ina"}
    assert get_loc_varlist(section) == ["e_pas", "i_pas", "i_membrane_extracellular", "v"]
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
    sim = SimpleNamespace(neuron=SimpleNamespace(h=SimpleNamespace(Vector=Vector, _ref_t="time")))
    recording = FixedDtRecordingCustom("voltage", location=location)

    recording.instantiate(sim=sim, icell=object())

    assert calls == [(0.1,), (0.1,)]
