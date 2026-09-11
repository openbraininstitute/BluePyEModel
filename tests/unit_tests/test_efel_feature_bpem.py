"""Tests for eFEL feature helpers."""

import numpy
import pytest

from bluepyemodel.evaluation.efel_feature_bpem import (
    DendFitFeature,
    DendFitMultiProtocolsFeature,
    eFELFeatureBPEM,
)


def test_bpo_feature_and_score():
    feature = eFELFeatureBPEM(
        "threshold",
        efel_feature_name="bpo_threshold_current",
        exp_mean=1.0,
        exp_std=0.5,
    )

    assert feature.calculate_bpo_feature({"bpo_threshold_current": 2.0}) == 2.0
    assert feature.calculate_bpo_score({"bpo_threshold_current": 2.0}) == 2.0
    assert feature.calculate_score({}) == feature.max_score


def test_construct_efel_trace_handles_current_and_callable_bounds():
    feature = eFELFeatureBPEM(
        "feature",
        efel_feature_name="voltage_base",
        recording_names={"": "step.soma.v", "dend": "step.dend.v"},
        stim_start=lambda: 1.0,
        stim_end=lambda: 2.0,
    )
    responses = {
        "step.soma.v": {"time": [0, 1], "voltage": [-80, -79]},
        "step.dend.v": {"time": [0, 1], "voltage": [-81, -80]},
        "step.iclamp.i": {"voltage": [0.1, 0.2]},
    }

    trace = feature._construct_efel_trace(responses)

    assert trace["T"] == [0, 1]
    assert trace["V;dend"] == [-81, -80]
    assert trace["I"] == [0.1, 0.2]
    assert trace["stim_start;dend"] == [1.0]


def test_construct_efel_trace_requires_soma_recording():
    feature = eFELFeatureBPEM(
        "feature", efel_feature_name="voltage_base", recording_names={"dend": "dend.v"}
    )

    with pytest.raises(ValueError, match="needs to be in recording_names"):
        feature._construct_efel_trace({})


def test_dend_fit_feature_linear_fit_and_properties():
    feature = DendFitFeature(
        "fit",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names={"": "soma.v", "10": "dend10.v"},
        linear=True,
    )

    slope = feature.fit([0, 10], [2.0, 1.0])

    assert slope == pytest.approx(-0.1, abs=1e-3)
    assert feature.distances == [0, 10]
    assert list(feature.recording_names_list) == ["soma.v", "dend10.v"]
    with pytest.raises(NotImplementedError):
        _ = feature.locations


def test_dend_fit_multi_protocol_properties():
    feature = DendFitMultiProtocolsFeature(
        "fit",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names={"": "IDrestapical[050,100]_100.apical.v"},
    )

    assert feature.distances == [0, 50, 100]
    assert feature.locations == ["soma", "apical050", "apical100"]
    assert list(feature.recording_names_list) == [
        "IDrest_100.soma.v",
        "IDrestapical050_100.apical050.v",
        "IDrestapical100_100.apical100.v",
    ]


def test_calculate_score_caps_non_bpo_score():
    feature = eFELFeatureBPEM(
        "feature",
        efel_feature_name="voltage_base",
        recording_names={"": "soma.v"},
        exp_mean=0.0,
        exp_std=1.0,
    )
    feature.calculate_feature = lambda responses: numpy.array([1000.0])

    assert feature.calculate_score({}) == feature.max_score
