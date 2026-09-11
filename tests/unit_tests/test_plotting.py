"""Smoke tests for plotting helpers."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy

from bluepyemodel.emodel_pipeline.plotting import (
    EPSP_fit,
    bAP_fit,
    compute_attenuation,
    plot_traces_current,
    traces_title,
)


def test_bap_fit_and_epsp_fit():
    class Feature:
        def fit(self, distances, values):
            assert distances == [0, 10]
            assert values == [1, 0.5]
            return -0.1

        @staticmethod
        def exp_decay(x, slope):
            return x + slope[0]

        @staticmethod
        def exp(x, slope):
            return x * slope[0]

    feature = Feature()
    distances = [0, 10]
    values = [1, 0.5]

    bap_x, bap_y = bAP_fit(feature, distances, values, npoints=3)
    epsp_x, epsp_y = EPSP_fit(feature, distances, values, npoints=3)

    numpy.testing.assert_allclose(bap_x, [0, 5, 10])
    numpy.testing.assert_allclose(bap_y, [-0.1, 4.9, 9.9])
    numpy.testing.assert_allclose(epsp_x, [0, 5, 10])
    numpy.testing.assert_allclose(epsp_y, [0, -0.5, -1])


def test_compute_attenuation():
    class Feature:
        def __init__(self, distances, values):
            self.distances = distances
            self.values = values

        def get_distances_feature_values(self, responses):
            return self.distances, self.values

    distances, attenuation = compute_attenuation(
        Feature([10, 20], [2, 3]), Feature([10, 20], [1, 2]), {}
    )

    assert distances == [10, 20]
    numpy.testing.assert_allclose(attenuation, [2, 1.5])


def test_plot_traces_current_and_traces_title():
    figure, axis = plt.subplots()
    plot_traces_current(axis, [0, 1], [-0.1, 0.2])
    model = SimpleNamespace(
        emodel_metadata=SimpleNamespace(emodel="model", iteration="iteration"),
        seed=4,
    )

    assert axis.get_xlabel() == "Time (ms)"
    assert axis.get_ylabel() == "Stim Current (nA)"
    assert "Threshold current = 0.1000 nA" in traces_title(model, threshold=0.1)
    plt.close(figure)
