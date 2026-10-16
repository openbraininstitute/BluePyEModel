"""Synthetic coverage for voltage/current trace and waveform plots."""

import sys
from types import ModuleType, SimpleNamespace

import matplotlib
import numpy
import pytest

matplotlib.use("Agg")

from bluepyemodel.emodel_pipeline import plotting as plotting_module

from .plotting_fakes import (
    SyntheticCell,
    SyntheticProtocol,
    make_model,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    matplotlib.pyplot.close("all")


def test_traces_title_includes_all_optional_values():
    model = make_model()

    title = plotting_module.traces_title(
        model, threshold=0.2, holding=-0.05, rmp=-75.0, rin=150.0
    )

    assert "L5PC" in title
    assert "iteration = iter1 ; seed = 1" in title
    assert "Threshold current = 0.2000 nA" in title
    assert "Holding current = -0.0500 nA" in title
    assert "Resting membrane potential = -75.00 mV" in title
    assert "Input Resistance = 150.00 MOhm" in title


def test_traces_title_without_optional_values():
    title = plotting_module.traces_title(make_model())

    assert "Threshold current" not in title
    assert "Resting membrane potential" not in title


def test_plot_traces_current_sets_finite_limits():
    _, ax = matplotlib.pyplot.subplots()

    plotting_module.plot_traces_current(ax, [0.0, 1.0], numpy.array([0.0, 0.5]))

    assert ax.get_ylabel() == "Stim Current (nA)"
    assert ax.get_ylim() == pytest.approx((-0.2, 0.7))


def test_plot_traces_current_ignores_non_finite_limits():
    _, ax = matplotlib.pyplot.subplots()

    plotting_module.plot_traces_current(
        ax, [0.0, 1.0], numpy.array([numpy.nan, numpy.nan])
    )

    assert ax.get_ylabel() == "Stim Current (nA)"


def test_thumbnail_writes_png(tmp_path):
    model = make_model()
    responses = {"IDrest_100.soma.v": {"time": [0.0, 1.0], "voltage": [-80.0, -70.0]}}

    fig, ax = plotting_module.thumbnail(
        model, responses, {"IDrest_100.soma.v"}, figures_dir=tmp_path, write_fig=True
    )

    assert fig is not None
    assert ax.get_xlabel() == "Time (ms)"
    assert ax.get_ylabel() == "Voltage (mV)"
    numpy.testing.assert_allclose(ax.lines[0].get_ydata(), [-80.0, -70.0])
    assert (tmp_path / "L5PC__1__thumbnail.png").is_file()


def test_thumbnail_returns_none_when_response_missing(tmp_path):
    model = make_model()

    fig, ax = plotting_module.thumbnail(
        model, {}, {"IDrest_100.soma.v"}, figures_dir=tmp_path, write_fig=False
    )

    assert fig is None
    assert ax is None


def test_traces_returns_none_without_trace_names(tmp_path):
    fig, axs = plotting_module.traces(
        make_model(),
        {"bpo_threshold_current": 0.2},
        set(),
        figures_dir=tmp_path,
        write_fig=False,
    )

    assert fig is None
    assert axs is None


def test_traces_plots_voltage_and_stimulus_current(tmp_path):
    model = make_model()
    responses = {
        "IDrest_100.soma.v": {"time": [0.0, 1.0], "voltage": [-80.0, -70.0]},
        "bpo_threshold_current": 0.2,
        "bpo_holding_current": -0.05,
        "bpo_rmp": -75.0,
        "bpo_rin": 150.0,
    }
    stimulus = SimpleNamespace(generate=lambda: ([0.0, 1.0], [0.0, 0.2]))
    stimuli = {"IDrest_100": SimpleNamespace(stimulus=stimulus)}

    fig, axs = plotting_module.traces(
        model,
        responses,
        {"IDrest_100.soma.v"},
        stimuli=stimuli,
        figures_dir=tmp_path,
        write_fig=True,
    )

    assert fig is not None
    assert axs[0, 0].get_title() == "IDrest_100.soma.v"
    assert (tmp_path / "L5PC__1__traces.pdf").is_file()


def test_traces_skips_empty_stimulus_waveform(tmp_path):
    """An empty stimulus waveform must not add a twinned current axis."""
    model = make_model()
    responses = {"IDrest_100.soma.v": {"time": [0.0, 1.0], "voltage": [-80.0, -70.0]}}
    empty = {
        "IDrest_100": SimpleNamespace(
            stimulus=SimpleNamespace(generate=lambda: ([], []))
        )
    }
    populated = {
        "IDrest_100": SimpleNamespace(
            stimulus=SimpleNamespace(generate=lambda: ([0.0, 1.0], [0.0, 0.2]))
        )
    }

    fig_empty, axs_empty = plotting_module.traces(
        model,
        responses,
        {"IDrest_100.soma.v"},
        stimuli=empty,
        figures_dir=tmp_path,
        write_fig=False,
    )
    fig_populated, _ = plotting_module.traces(
        model,
        responses,
        {"IDrest_100.soma.v"},
        stimuli=populated,
        figures_dir=tmp_path,
        write_fig=False,
    )

    # The voltage trace is still drawn on the primary axis.
    assert axs_empty[0, 0].get_ylabel() == "Voltage (mV)"
    assert len(axs_empty[0, 0].get_lines()) == 1

    # The populated waveform adds exactly one twinned current axis; the empty
    # waveform adds none, so the empty figure has one fewer axes.
    assert len(fig_populated.axes) == len(fig_empty.axes) + 1
    assert all(ax.get_ylabel() != "Stim Current (nA)" for ax in fig_empty.axes)
    assert any(ax.get_ylabel() == "Stim Current (nA)" for ax in fig_populated.axes)


def test_traces_handles_falsy_response(tmp_path):
    model = make_model()
    responses = {"IDrest_100.soma.v": {}}

    fig, axs = plotting_module.traces(
        model, responses, {"IDrest_100.soma.v"}, figures_dir=tmp_path, write_fig=False
    )

    assert fig is not None
    assert axs[0, 0].get_title() == "IDrest_100.soma.v"


def test_phase_plot_uses_custom_extraction_and_model_response(tmp_path, monkeypatch):
    model = make_model(seed=3)
    model.responses = {
        "IDrest_100.soma.v": {
            "time": numpy.array([0.0, 1.0, 2.0]),
            "voltage": numpy.array([-80.0, -65.0, -75.0]),
        }
    }
    monkeypatch.setattr(
        plotting_module, "read_extraction_output", lambda _: [SyntheticCell()]
    )

    plotting_module.phase_plot(
        [model],
        tmp_path,
        ["idrest"],
        100.0,
        2.0,
        custom_bluepyefe_cells_pklpath="cells.pkl",
        write_fig=True,
    )

    assert (tmp_path / "L5PC__3__phase_plot.pdf").is_file()


def test_plot_trace_comparison_plots_experiment_and_model(tmp_path, monkeypatch):
    model = make_model(seed=4)
    model.responses = {
        "RMPProtocol.soma.v": {
            "time": numpy.array([0.0, 1.0]),
            "voltage": numpy.array([-80.0, -70.0]),
        },
        "RinProtocol.soma.v": {
            "time": numpy.array([0.0, 1.0]),
            "voltage": numpy.array([-80.0, -60.0]),
        },
        "IDrest_100.soma.v": {
            "time": numpy.array([0.0, 1.0]),
            "voltage": numpy.array([-80.0, -65.0]),
        },
    }
    monkeypatch.setattr(
        plotting_module, "read_extraction_output", lambda _: [SyntheticProtocol()]
    )

    plotting_module.plot_trace_comparison(
        [model], tmp_path, custom_bluepyefe_protocols_pklpath="protocols.pkl"
    )

    assert (tmp_path / "L5PC__4__trace_comparison.pdf").is_file()


def test_plot_sinespec_skips_missing_responses(tmp_path):
    fig, axs = plotting_module.plot_sinespec(
        make_model(), {}, {"amp": 150}, {}, figures_dir=tmp_path, write_fig=False
    )

    assert fig is not None
    assert len(axs) == 3
    # All three panels are created but left empty when the response is absent.
    assert all(not axis.lines for axis in axs)


def test_plot_sinespec_plots_current_voltage_and_impedance(monkeypatch, tmp_path):
    monkeypatch.setattr(
        plotting_module,
        "get_impedance",
        lambda *args: (numpy.array([1.0, 2.0]), numpy.array([0.5, 0.25])),
    )
    responses = {
        "SineSpec_100.iclamp.i": {"time": [0, 1], "voltage": [0.1, 0.2]},
        "SineSpec_100.soma.v": {"time": [0, 1], "voltage": [-80, -70]},
    }

    _, axes = plotting_module.plot_sinespec(
        make_model(), responses, {"amp": 100}, {}, figures_dir=tmp_path, write_fig=False
    )

    assert [axis.get_ylabel() for axis in axes] == [
        "Injected current (nA)",
        "Voltage (mV)",
        "normalized Z",
    ]
    assert all(len(axis.lines) == 1 for axis in axes)


def test_currentscape_requires_responses_or_output_directory():
    with pytest.raises(TypeError, match="Responses or output directory"):
        plotting_module.currentscape()


def test_currentscape_passes_ordered_response_data_to_optional_package(
    tmp_path, monkeypatch
):
    calls = {}
    currentscape_package = ModuleType("currentscape")
    currentscape_module = ModuleType("currentscape.currentscape")

    def fake_plot(voltage, currents, config, ions_data, time):
        calls.update(
            voltage=voltage,
            currents=currents,
            config=config,
            ions_data=ions_data,
            time=time,
        )
        return matplotlib.pyplot.figure()

    currentscape_module.plot_currentscape = fake_plot
    monkeypatch.setitem(sys.modules, "currentscape", currentscape_package)
    monkeypatch.setitem(sys.modules, "currentscape.currentscape", currentscape_module)
    monkeypatch.setattr(
        plotting_module,
        "get_ordered_currentscape_keys",
        lambda keys: {
            "Step_100": {
                "soma": {
                    "voltage_key": "Step_100.soma.v",
                    "current_keys": ["Step_100.soma.ina"],
                    "current_names": ["ina"],
                    "ion_conc_keys": ["Step_100.soma.cai"],
                    "ion_conc_names": ["cai"],
                }
            }
        },
    )
    responses = {
        "Step_100.soma.v": {"time": [0.0, 1.0], "voltage": [-80.0, -70.0]},
        "Step_100.soma.ina": {"time": [0.0, 1.0], "voltage": [0.1, 0.2]},
        "Step_100.soma.cai": {"time": [0.0, 1.0], "voltage": [0.01, 0.02]},
    }
    config = {}

    plotting_module.currentscape(
        responses=responses,
        config=config,
        metadata_str="L5PC__1",
        figures_dir=tmp_path,
        emodel="L5PC",
    )

    assert calls["voltage"] == [-80.0, -70.0]
    assert calls["currents"] == [[0.1, 0.2]]
    assert calls["ions_data"] == [[0.01, 0.02]]
    assert calls["config"]["output"]["savefig"] is True
    assert calls["config"]["output"]["fname"] == "L5PC__1__currentscape.Step_100.soma"
    assert config == {}
