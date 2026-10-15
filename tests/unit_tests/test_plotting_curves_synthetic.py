"""Synthetic coverage for IV/FI curves and dendritic attenuation plots."""

from types import SimpleNamespace

import matplotlib
import numpy
import pytest

matplotlib.use("Agg")

from bluepyemodel.emodel_pipeline import plotting as plotting_module

from .plotting_fakes import make_dend_feature, make_model


@pytest.fixture(autouse=True)
def close_figures():
    yield
    matplotlib.pyplot.close("all")


def test_bAP_fit_returns_decay_curve():
    feature = make_dend_feature(decay=True)

    x_fit, y_fit = plotting_module.bAP_fit(feature, [0, 100], [2.0, 1.0], npoints=5)

    assert len(x_fit) == 5
    assert len(y_fit) == 5
    assert x_fit[0] == pytest.approx(0.0)
    assert x_fit[-1] == pytest.approx(100.0)
    assert y_fit[0] > y_fit[-1]


def test_EPSP_fit_returns_growth_curve():
    feature = make_dend_feature()

    x_fit, y_fit = plotting_module.EPSP_fit(feature, [0, 100], [1.0, 2.0], npoints=4)

    assert len(x_fit) == 4
    assert y_fit[-1] > y_fit[0]


def test_compute_attenuation_returns_ratios():
    dend_feature = SimpleNamespace(
        get_distances_feature_values=lambda responses: ([0, 100], [0.5, 0.25])
    )
    soma_feature = SimpleNamespace(
        get_distances_feature_values=lambda responses: ([0], [1.0])
    )

    distances, attenuation = plotting_module.compute_attenuation(
        dend_feature, soma_feature, {}
    )

    numpy.testing.assert_allclose(distances, [0, 100])
    numpy.testing.assert_allclose(attenuation, [0.5, 0.25])


def test_dendritic_feature_plot_covers_model_fit_and_invalid_name(
    tmp_path, monkeypatch
):
    feature = SimpleNamespace(
        name="ISI_CV_linear",
        get_distances_feature_values=lambda responses: ([0.0, 100.0], [1.0, 2.0]),
        fit=lambda distances, values: [0.01, 1.0],
        linear_fit=lambda x, slope: numpy.asarray(x) * slope[0][0] + slope[0][1],
    )
    monkeypatch.setattr(
        plotting_module,
        "read_dendritic_data",
        lambda _: ([0.0, 100.0, 200.0], [1.0, 1.5, 2.0]),
    )

    fig, ax = plotting_module.dendritic_feature_plot(
        make_model(), {}, feature, "ISI_CV", tmp_path, write_fig=True
    )

    assert fig is not None
    assert len(ax.lines) == 2
    assert (tmp_path / "L5PC__1__ISI_CV_linear.pdf").is_file()
    with pytest.raises(ValueError, match="Expected 'ISI_CV' or 'rheobase'"):
        plotting_module.dendritic_feature_plot(
            make_model(), {}, feature, "other", tmp_path
        )


def test_plot_iv_curves_classifies_simulated_features(tmp_path, monkeypatch):
    model = make_model(seed=5)
    model.responses = {"bpo_threshold_current": 0.2}
    evaluator = SimpleNamespace(
        fitness_calculator=SimpleNamespace(
            calculate_values=lambda responses: {
                "iv_100.soma.v.maximum_voltage_from_voltagebase": -55.0,
                "iv_-100.soma.v.voltage_deflection_vb_ssse": -10.0,
            }
        )
    )
    calls = {}
    monkeypatch.setattr(
        plotting_module,
        "fill_in_IV_curve_evaluator",
        lambda evaluator, settings, prot_name, amps: evaluator,
    )
    monkeypatch.setattr(
        plotting_module, "compute_responses", lambda *args, **kwargs: [model]
    )
    monkeypatch.setattr(plotting_module, "read_extraction_output", lambda _: object())
    monkeypatch.setattr(
        plotting_module,
        "extract_experimental_data_for_IV_curve",
        lambda cells, settings, prot_name, n_bin: (
            {
                "amp_rel": [-100, 100],
                "feat_rel": [-70, -50],
                "feat_rel_err": [1, 1],
                "amp": [-0.1, 0.1],
                "feat_abs": [-70, -50],
                "feat_abs_err": [1, 1],
            },
            {
                "amp_rel": [-100, 100],
                "feat_rel": [-20, -5],
                "feat_rel_err": [1, 1],
                "amp": [-0.1, 0.1],
                "feat_abs": [-20, -5],
                "feat_abs_err": [1, 1],
            },
        ),
    )
    monkeypatch.setattr(
        plotting_module,
        "get_amplitude_from_feature_key",
        lambda key: 100 if "100" in key else -100,
    )
    monkeypatch.setattr(
        plotting_module, "rel_to_abs_amplitude", lambda amp, responses: amp / 100.0
    )
    monkeypatch.setattr(
        plotting_module,
        "save_fig",
        lambda directory, name: calls.setdefault("figure", (directory, name)),
    )

    plotting_module.plot_IV_curves(
        evaluator,
        [model],
        SimpleNamespace(),
        tmp_path,
        {},
        map,
        None,
        custom_bluepyefe_cells_pklpath="cells.pkl",
        write_fig=True,
        n_bin=1,
    )

    assert calls["figure"][1] == "L5PC__5__IV_curve.pdf"


def test_plot_fi_curves_comparison_updates_and_plots(tmp_path, monkeypatch):
    model = make_model(seed=6)
    model.responses = {"idrest_100.soma.v": {"time": [0], "voltage": [-70]}}
    execution = []
    main_protocol = SimpleNamespace(
        compute_execution_order=lambda: execution.append(True) or ["idrest"]
    )
    evaluator = SimpleNamespace(fitness_protocols={"main_protocol": main_protocol})
    plotted = {}
    monkeypatch.setattr(plotting_module, "read_extraction_output", lambda _: object())
    monkeypatch.setattr(
        plotting_module,
        "get_experimental_FI_curve_for_plotting",
        lambda cells, prot_name, n_bin: ([100], [5], [0], [0.1], [5], [0]),
    )
    monkeypatch.setattr(
        plotting_module, "get_original_protocol_name", lambda name, ev: name
    )
    monkeypatch.setattr(plotting_module, "update_evaluator", lambda *args: evaluator)
    monkeypatch.setattr(plotting_module, "compute_responses", lambda *args: [model])
    monkeypatch.setattr(
        plotting_module,
        "get_simulated_FI_curve_for_plotting",
        lambda ev, responses, name: ([-0.1], [2]),
    )
    monkeypatch.setattr(
        plotting_module,
        "plot_fi_curves",
        lambda experimental, simulated, directory, model, write: plotted.update(
            experimental=experimental, simulated=simulated
        ),
    )

    plotting_module.plot_FI_curves_comparison(
        evaluator,
        [model],
        SimpleNamespace(),
        None,
        map,
        tmp_path,
        "idrest",
        custom_bluepyefe_cells_pklpath="cells.pkl",
        write_fig=False,
        n_bin=1,
    )

    assert execution == [True]
    assert plotted["experimental"][0] == [100]
    assert plotted["simulated"] == ([-0.1], [2])


def test_plot_bap_covers_valid_and_invalid_feature_data(tmp_path, monkeypatch):
    valid_apical = SimpleNamespace(
        get_distances_feature_values=lambda responses: ([0.0, 100.0], [2.0, 1.0])
    )
    valid_basal = SimpleNamespace(
        get_distances_feature_values=lambda responses: ([0.0, 80.0], [1.5, 0.8])
    )
    monkeypatch.setattr(
        plotting_module,
        "bAP_fit",
        lambda feature, distances, values: ([0.0, 100.0], [values[0], values[-1]]),
    )

    fig, ax = plotting_module.plot_bAP(
        make_model(), {}, valid_apical, valid_basal, tmp_path, write_fig=False
    )

    assert fig is not None
    assert len(ax.lines) == 2
    missing = SimpleNamespace(
        get_distances_feature_values=lambda responses: ([0.0], None)
    )
    _, missing_ax = plotting_module.plot_bAP(
        make_model(), {}, missing, valid_basal, tmp_path, write_fig=False
    )
    assert len(missing_ax.lines) == 0


def test_plot_bap_returns_empty_axis_for_mismatched_features(tmp_path):
    mismatch = SimpleNamespace(
        get_distances_feature_values=lambda responses: ([0.0, 1.0], [1.0])
    )
    valid = SimpleNamespace(
        get_distances_feature_values=lambda responses: ([0.0], [1.0])
    )

    _, ax = plotting_module.plot_bAP(
        make_model(), {}, mismatch, valid, tmp_path, write_fig=False
    )

    assert len(ax.lines) == 0


def test_plot_epsilon_s_p_covers_valid_and_mismatched_data(tmp_path, monkeypatch):
    def feature(values):
        return SimpleNamespace(
            get_distances_feature_values=lambda responses: ([0.0, 100.0], values)
        )

    monkeypatch.setattr(
        plotting_module,
        "EPSP_fit",
        lambda feature_obj, distances, values: ([0.0, 100.0], [values[0], values[-1]]),
    )
    fig, ax = plotting_module.plot_EPSP(
        make_model(),
        {},
        feature([2.0, 1.0]),
        feature([1.0, 1.0]),
        feature([1.5, 0.8]),
        feature([1.0, 1.0]),
        tmp_path,
        write_fig=False,
    )

    assert fig is not None
    assert len(ax.lines) == 2
    mismatch = SimpleNamespace(
        get_distances_feature_values=lambda responses: ([0.0, 1.0], [1.0])
    )
    _, mismatch_ax = plotting_module.plot_EPSP(
        make_model(),
        {},
        mismatch,
        feature([1.0, 1.0, 1.0]),
        feature([1.0, 1.0]),
        feature([1.0, 1.0]),
        tmp_path,
        write_fig=False,
    )
    assert len(mismatch_ax.lines) == 0


def test_dendritic_feature_plots_dispatches_and_normalizes_stimulus(
    tmp_path, monkeypatch
):
    calls = []
    feature = SimpleNamespace(name="ISI_CV_linear", stimulus_current=None)
    other = SimpleNamespace(name="unrelated", stimulus_current=lambda: None)
    model = make_model()
    model.responses = {}
    model.evaluator = SimpleNamespace(
        fitness_calculator=SimpleNamespace(
            objectives=[
                SimpleNamespace(features=[feature]),
                SimpleNamespace(features=[other]),
            ]
        )
    )
    monkeypatch.setattr(
        plotting_module,
        "dendritic_feature_plot",
        lambda *args, **kwargs: calls.append(args),
    )

    plotting_module.dendritic_feature_plots(model, "ISI_CV", "all", tmp_path)

    assert feature.stimulus_current == 0.0
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Expected 'ISI_CV' or 'rheobase'"):
        plotting_module.dendritic_feature_plots(model, "unknown", "all", tmp_path)


def test_fi_curve_returns_axes_when_threshold_data_is_missing():
    """Without a threshold current the panels are created but stay unlabelled."""
    fig, axes = plotting_module.FI_curve(
        make_model(), {"bpo_holding_current": 0.1}, object(), write_fig=False
    )

    assert fig is not None
    assert len(axes) == 2
    assert axes[0].get_xlabel() == ""
