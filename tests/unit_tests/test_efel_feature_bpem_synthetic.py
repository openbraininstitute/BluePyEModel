import sys

import numpy
import pytest

import bluepyemodel.evaluation.efel_feature_bpem as feature_module
from bluepyemodel.evaluation.efel_feature_bpem import DendFitFeature, eFELFeatureBPEM


class FakeEfel:
    def __init__(self, values=None):
        self.values = values
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def set_threshold(self, value):
        self.calls.append(("threshold", value))

    def set_double_setting(self, name, value):
        self.calls.append(("double", name, value))

    def set_int_setting(self, name, value):
        self.calls.append(("int", name, value))

    def set_str_setting(self, name, value):
        self.calls.append(("string", name, value))

    def get_feature_values(self, traces, names, raise_warnings=False):
        self.calls.append(("get", traces, names, raise_warnings))
        return self.values


def base_responses():
    return {
        "soma.v": {"time": [0, 1], "voltage": [-80, -79]},
        "dend.v": {"time": [0, 1], "voltage": [-81, -80]},
        "step.iclamp.i": {"voltage": [0.1, 0.2]},
    }


def test_base_trace_handles_missing_recording_and_none_response():
    feature = eFELFeatureBPEM(
        "feature",
        efel_feature_name="voltage_base",
        recording_names={"": "soma.v", "dend": "dend.v"},
        stim_start=1.0,
        stim_end=2.0,
    )

    assert feature._construct_efel_trace({"soma.v": base_responses()["soma.v"]}) is None
    responses = base_responses()
    responses["dend.v"] = None
    assert feature._construct_efel_trace(responses) is None
    trace = feature._construct_efel_trace(base_responses())
    assert trace["stim_start"] == [1.0]
    assert trace["stim_end;dend"] == [2.0]


def test_setup_efel_applies_all_settings(monkeypatch):
    fake = FakeEfel()
    monkeypatch.setitem(sys.modules, "efel", fake)
    feature = eFELFeatureBPEM(
        "feature",
        threshold=-30.0,
        stimulus_current=lambda: 0.5,
        interp_step=0.1,
        double_settings={"double": 1.0},
        int_settings={"integer": 2},
        string_settings={"string": "value"},
    )

    feature._setup_efel()

    assert fake.calls == [
        ("reset",),
        ("threshold", -30.0),
        ("double", "stimulus_current", 0.5),
        ("double", "interp_step", 0.1),
        ("double", "double", 1.0),
        ("int", "integer", 2),
        ("string", "string", "value"),
    ]


def test_calculate_feature_uses_fake_efel_and_forwards_warnings(monkeypatch):
    fake = FakeEfel(values=[{"voltage_base": [-79.5]}])
    monkeypatch.setitem(sys.modules, "efel", fake)
    feature = eFELFeatureBPEM(
        "feature",
        efel_feature_name="voltage_base",
        recording_names={"": "soma.v"},
        stim_start=1.0,
        stim_end=2.0,
    )

    result = feature.calculate_feature(
        {"soma.v": base_responses()["soma.v"]}, raise_warnings=True
    )

    assert result == [-79.5]
    assert fake.calls[-2][0] == "get"
    assert fake.calls[-2][3] is True
    assert fake.calls[-1] == ("reset",)


def test_calculate_feature_bpo_and_missing_trace(monkeypatch):
    bpo = eFELFeatureBPEM("bpo", efel_feature_name="bpo_value")
    assert bpo.calculate_feature({"bpo_value": 3.0}) == [3.0]
    assert bpo.calculate_feature({})[0] is None

    fake = FakeEfel(values=[])
    monkeypatch.setitem(sys.modules, "efel", fake)
    feature = eFELFeatureBPEM(
        "feature", efel_feature_name="voltage_base", recording_names={"": "soma.v"}
    )
    assert feature.calculate_feature({}) is None
    assert fake.calls == []


def test_scores_cover_empty_none_nan_and_normal_values(monkeypatch):
    feature = eFELFeatureBPEM(
        "feature", efel_feature_name="voltage_base", exp_mean=2.0, exp_std=2.0
    )
    feature.calculate_feature = lambda responses: numpy.array([1.0, 3.0])
    assert feature.calulate_score_({}) == pytest.approx(0.5)
    feature.calculate_feature = lambda responses: None
    assert feature.calculate_score({}) == feature.max_score
    feature.calculate_feature = lambda responses: numpy.array([])
    assert feature.calculate_score({}) == feature.max_score
    feature.calculate_feature = lambda responses: numpy.array([numpy.nan])
    assert feature.calculate_score({}) == feature.max_score

    no_expected = eFELFeatureBPEM("feature", efel_feature_name="voltage_base")
    assert no_expected.calulate_score_({}) == 0


def dend_responses():
    return {
        "soma.v": {"time": [0, 1], "voltage": [1.0, 2.0]},
        "dend10.v": {"time": [0, 1], "voltage": [0.5, 1.0]},
        "dend20.v": {"time": [0, 1], "voltage": [0.2, 0.4]},
    }


def test_dend_trace_and_fit_model_branches(monkeypatch):
    feature = DendFitFeature(
        "fit",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names={"": "soma.v", "10": "dend10.v"},
        stim_start=lambda: 1.0,
        stim_end=2.0,
    )
    assert feature._construct_efel_trace({}) is None
    responses = dend_responses()
    responses["dend10.v"] = None
    assert feature._construct_efel_trace(responses) is None
    traces = feature._construct_efel_trace(dend_responses())
    assert traces[1]["stim_start"] == [1.0]
    assert traces[1]["stim_end"] == [2.0]

    feature.ymult = 2.0
    assert feature.exp_decay(numpy.array([0.0, 10.0]), 50.0)[0] == pytest.approx(2.0)
    assert feature.exp(10.0, 10.0) == pytest.approx(2.0 * numpy.e)
    assert feature.linear_fit(10.0, -0.1) == pytest.approx(1.0)

    calls = []

    def fake_curve_fit(model, distances, values, p0):
        calls.append((model, distances, values, p0))
        return numpy.array([3.5]), None

    monkeypatch.setattr(feature_module.opt, "curve_fit", fake_curve_fit)
    for kwargs in ({"linear": True}, {"decay": True}, {}):
        fitted = DendFitFeature(
            "fit",
            efel_feature_name="maximum_voltage_from_voltagebase",
            recording_names={"": "soma.v", "10": "dend10.v"},
            **kwargs,
        )
        assert fitted.fit([0, 10], [2.0, 1.0]) == 3.5
    assert calls[0][0].__name__ == "linear_fit"
    assert calls[1][0].__name__ == "exp_decay"
    assert calls[2][0].__name__ == "exp"


def test_dend_non_bpo_extraction_and_calculate_feature(monkeypatch):
    fake = FakeEfel(
        values=[
            {"maximum_voltage_from_voltagebase": [2.0]},
            {"maximum_voltage_from_voltagebase": None},
            {"maximum_voltage_from_voltagebase": [1.0]},
        ]
    )
    monkeypatch.setitem(sys.modules, "efel", fake)
    feature = DendFitFeature(
        "fit",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names={"": "soma.v", "10": "dend10.v", "20": "dend20.v"},
        exp_mean=1.5,
        exp_std=1.0,
        stim_start=1.0,
        stim_end=2.0,
    )
    distances, values = feature.get_distances_feature_values(
        dend_responses(), raise_warnings=True
    )
    assert distances == [0, 20]
    assert values == [2.0, 1.0]
    assert fake.calls[-1] == ("reset",)

    monkeypatch.setattr(feature, "fit", lambda distances, values: 4.0)
    assert feature.calculate_feature(dend_responses()) == [4.0]
    feature.recording_names = {"": "soma.v", "10": "missing.v"}
    assert feature.calculate_feature(dend_responses()) is None
    feature.recording_names = {"": "soma.v", "10": "dend10.v"}
    assert feature.get_distances_feature_values({}) == ([], None)


def test_dend_bpo_values_adjust_soma_holding_current():
    class LocatedDendFit(DendFitFeature):
        @property
        def locations(self):
            return ["soma", "apical10", "apical20"]

    feature = LocatedDendFit(
        "threshold",
        efel_feature_name="bpo_threshold_current",
        recording_names={"": "soma.v", "10": "dend10.v", "20": "dend20.v"},
    )
    responses = {
        "bpo_threshold_current": 1.0,
        "bpo_threshold_current_apical10": None,
        "bpo_threshold_current_apical20": 3.0,
        "bpo_holding_current": 0.25,
    }
    assert feature.get_distances_feature_values(responses) == ([0, 20], [1.25, 3.0])


def test_dend_score_and_bpo_missing_score():
    feature = DendFitFeature(
        "fit",
        efel_feature_name="maximum_voltage_from_voltagebase",
        exp_mean=1.0,
        exp_std=1.0,
    )
    feature.calculate_feature = lambda responses: numpy.array([numpy.nan])
    assert feature.calculate_score({}) == feature.max_score

    bpo = eFELFeatureBPEM(
        "bpo", efel_feature_name="bpo_value", exp_mean=2.0, exp_std=1.0
    )
    assert bpo.calculate_bpo_score({"bpo_value": 1.0}) == 1.0
