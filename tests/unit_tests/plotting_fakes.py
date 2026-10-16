"""Shared fakes for the synthetic ``emodel_pipeline.plotting`` tests.

These helpers build the minimal duck-typed stand-ins that the plotting module
consumes (models, metadata and dendritic-fit features), so the plotting tests
stay deterministic and independent of NEURON and of the access points.
"""

from types import SimpleNamespace

import numpy

from bluepyemodel.evaluation.efel_feature_bpem import DendFitFeature


def make_metadata(emodel="L5PC", iteration="iter1"):
    """Return a metadata stand-in exposing the attributes plotting reads."""
    return SimpleNamespace(
        emodel=emodel,
        iteration=iteration,
        as_string=lambda seed=None: f"{emodel}__{seed}" if seed is not None else emodel,
    )


def make_model(scores=None, scores_validation=None, parameters=None, seed=1):
    """Return a model stand-in with empty-by-default score/parameter mappings."""
    return SimpleNamespace(
        emodel_metadata=make_metadata(),
        seed=seed,
        scores=scores if scores is not None else {},
        scores_validation=scores_validation if scores_validation is not None else {},
        parameters=parameters if parameters is not None else {},
    )


def make_dend_feature(linear=None, decay=None):
    """Return a ``DendFitFeature`` configured for a soma/dendrite recording pair."""
    return DendFitFeature(
        "fit",
        efel_feature_name="maximum_voltage_from_voltagebase",
        recording_names={"": "soma.v", "100": "dend100.v"},
        linear=linear,
        decay=decay,
    )


class SyntheticRecording:
    """A BluePyEfe-style recording with a fixed three-point voltage trace."""

    def __init__(self, protocol_name="IDrest", amp_rel=100.0, amp=0.1):
        self.protocol_name = protocol_name
        self.amp_rel = amp_rel
        self.amp = amp
        self.t = numpy.array([0.0, 1.0, 2.0])
        self.time = self.t
        self.voltage = numpy.array([-80.0, -70.0, -75.0])


class SyntheticCell:
    """A cell exposing a single default recording."""

    def __init__(self):
        self.recordings = [SyntheticRecording()]


class SyntheticProtocol:
    """A protocol whose single recording echoes its name and amplitude."""

    def __init__(self, name="IDrest", amplitude=-40.0):
        self.name = name
        self.amplitude = amplitude
        self.recordings = [SyntheticRecording(protocol_name=name, amp=amplitude)]
