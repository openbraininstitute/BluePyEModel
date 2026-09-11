"""Synthetic coverage for score, distribution and optimisation reports."""

from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")

from bluepyemodel.emodel_pipeline import plotting as plotting_module

from .plotting_fakes import make_metadata, make_model


@pytest.fixture(autouse=True)
def close_figures():
    yield
    matplotlib.pyplot.close("all")


def test_scores_writes_figure_and_flags_high_scores(tmp_path):
    model = make_model(
        scores={"Step_150.soma.v.mean_frequency": 7.5},
        scores_validation={"IV_-40.soma.v.voltage_base": 1.0},
    )

    fig, axs = plotting_module.scores(model, figures_dir=tmp_path, write_fig=True)

    assert fig is not None
    assert axs[0, 0].get_xlabel() == "z-score"
    assert (tmp_path / "L5PC__1__scores.pdf").is_file()


def test_parameters_distribution_writes_figure(tmp_path):
    models = [
        make_model(parameters={"gNa": 0.5, "gK": 0.2}),
        make_model(parameters={"gNa": 0.7, "gK": 0.4}),
    ]
    lbounds = {"gNa": 0.0, "gK": 0.0}
    ubounds = {"gNa": 1.0, "gK": 1.0}

    fig, axs = plotting_module.parameters_distribution(
        models, lbounds, ubounds, figures_dir=tmp_path, write_fig=True
    )

    assert fig is not None
    assert axs[0, 0].get_xticks().tolist() == [-1, 0, 1]
    assert (tmp_path / "L5PC__parameters_distribution.pdf").is_file()


def test_parameters_distribution_warns_on_mixed_etypes(tmp_path, caplog):
    models = [
        SimpleNamespace(
            emodel_metadata=make_metadata(emodel="A"), seed=1, parameters={"gNa": 0.5}
        ),
        SimpleNamespace(
            emodel_metadata=make_metadata(emodel="B"), seed=2, parameters={"gNa": 0.6}
        ),
    ]

    plotting_module.parameters_distribution(
        models, {"gNa": 0.0}, {"gNa": 1.0}, figures_dir=tmp_path, write_fig=False
    )

    assert "More than one e-type" in caplog.text


def test_optimisation_plots_progress(tmp_path, monkeypatch):
    logbook = SimpleNamespace(
        select=lambda field: {
            "gen": [0, 1, 2],
            "min": [5.0, 3.0, 1.0],
            "avg": [6.0, 4.0, 2.0],
        }[field]
    )
    monkeypatch.setattr(
        plotting_module,
        "read_checkpoint",
        lambda path: ({"logbook": logbook, "generation": 2}, 1),
    )

    fig, axs = plotting_module.optimisation(
        optimiser="IBEA",
        emodel="L5PC",
        iteration="iter1",
        seed=1,
        checkpoint_path=str(tmp_path / "chk.pkl"),
        figures_dir=tmp_path,
        write_fig=True,
    )

    assert fig is not None
    assert axs[0, 0].get_xlabel() == "Number of generations"
    assert (tmp_path / "chk__optimisation.pdf").is_file()


def test_optimisation_reports_cma_completion(tmp_path, monkeypatch):
    logbook = SimpleNamespace(
        select=lambda field: {"gen": [0, 1], "min": [5.0, 1.0], "avg": [6.0, 2.0]}[
            field
        ]
    )
    monkeypatch.setattr(
        plotting_module,
        "read_checkpoint",
        lambda path: (
            {
                "logbook": logbook,
                "generation": 1,
                "CMA_es": SimpleNamespace(active=False),
            },
            1,
        ),
    )

    fig, axs = plotting_module.optimisation(
        optimiser="SO-CMA",
        emodel="L5PC",
        iteration=None,
        seed=None,
        checkpoint_path=str(tmp_path / "chk.pkl"),
        figures_dir=tmp_path,
        write_fig=False,
    )

    assert fig is not None
    assert "is finished: True" in axs[0, 0].get_legend().get_title().get_text()


def test_plot_models_returns_empty_without_models(tmp_path, monkeypatch):
    metadata = make_metadata()
    access_point = SimpleNamespace(
        emodel_metadata=metadata,
        get_emodels=lambda *args: [],
    )
    evaluator = SimpleNamespace(
        fitness_protocols={"main_protocol": SimpleNamespace(protocols={})}
    )

    result = plotting_module.plot_models(
        access_point,
        mapper=map,
        figures_dir=tmp_path,
        cell_evaluator=evaluator,
        plot_optimisation_progress=False,
        plot_parameter_evolution=False,
        plot_distributions=False,
        plot_scores=False,
        plot_traces=False,
        plot_thumbnail=False,
        plot_dendritic_ISI_CV=False,
        plot_dendritic_rheobase=False,
    )

    assert result == []


def test_evolution_parameters_density_plots_checkpoint_histograms(
    tmp_path, monkeypatch
):
    evaluator = SimpleNamespace(
        params=[
            SimpleNamespace(bounds=(0.0, 1.0)),
            SimpleNamespace(bounds=(-1.0, 1.0)),
        ],
        param_names=["gNa", "gK"],
    )
    history = SimpleNamespace(
        genealogy_history={
            1: [0.2, -0.4],
            2: [0.4, 0.0],
            3: [0.6, 0.4],
            4: [0.8, 0.8],
        }
    )
    runs = {
        "early.pkl": ({"generation": 2, "history": history, "population": [1, 2]}, 1),
        "late.pkl": ({"generation": 8, "history": history, "population": [1, 2]}, 2),
    }
    monkeypatch.setattr(plotting_module, "read_checkpoint", runs.__getitem__)

    fig, axs = plotting_module.evolution_parameters_density(
        evaluator,
        ["early.pkl", "late.pkl"],
        make_metadata(),
        figures_dir=tmp_path,
        write_fig=True,
    )

    assert fig is not None
    assert len(axs) == 5
    assert (tmp_path / "L5PC__all_seeds__evo_parameter_density.pdf").is_file()


def test_plot_models_dispatches_score_plot(tmp_path, monkeypatch):
    model = make_model(seed=7)
    access_point = SimpleNamespace(
        emodel_metadata=make_metadata(), get_emodels=lambda *args: [model]
    )
    evaluator = SimpleNamespace(
        fitness_protocols={"main_protocol": SimpleNamespace(protocols={})}
    )
    calls = []
    monkeypatch.setattr(
        plotting_module,
        "scores",
        lambda model, directory: calls.append((model, directory)),
    )

    result = plotting_module.plot_models(
        access_point,
        map,
        figures_dir=tmp_path,
        cell_evaluator=evaluator,
        plot_optimisation_progress=False,
        plot_parameter_evolution=False,
        plot_distributions=False,
        plot_scores=True,
        plot_traces=False,
        plot_thumbnail=False,
        plot_dendritic_ISI_CV=False,
        plot_dendritic_rheobase=False,
    )

    assert result == [model]
    assert calls == [(model, tmp_path / "scores" / "all")]


def test_plot_models_dispatches_all_optional_plotters(tmp_path, monkeypatch):
    model = make_model(seed=8)
    model.passed_validation = True
    model.responses = {"Step.soma.v": {"time": [0.0], "voltage": [-70.0]}}
    metadata = make_metadata()
    pipeline_settings = SimpleNamespace(
        currentscape_config=None,
        efel_settings={},
        phase_plot_settings={
            "prot_names": ["Step"],
            "amplitude": 100.0,
            "amp_window": 5.0,
            "relative_amp": True,
        },
    )
    access_point = SimpleNamespace(
        emodel_metadata=metadata,
        pipeline_settings=pipeline_settings,
        get_fitness_calculator_configuration=lambda: SimpleNamespace(protocols=[]),
    )
    evaluator = SimpleNamespace(
        fitness_protocols={"main_protocol": SimpleNamespace(protocols={})},
        cell_model=SimpleNamespace(params={}),
    )
    calls = []
    access_point.get_emodels = lambda *args: [model]
    monkeypatch.setattr(plotting_module, "compute_responses", lambda *a, **k: [model])
    monkeypatch.setattr(
        plotting_module, "existing_checkpoint_paths", lambda _: ["seed=8.pkl"]
    )
    monkeypatch.setattr(
        plotting_module, "optimisation", lambda *a, **k: calls.append("optimisation")
    )
    monkeypatch.setattr(
        plotting_module,
        "evolution_parameters_density",
        lambda *a, **k: calls.append("evolution"),
    )
    monkeypatch.setattr(
        plotting_module,
        "parameters_distribution",
        lambda *a, **k: calls.append("distribution"),
    )
    monkeypatch.setattr(
        plotting_module, "scores", lambda *a, **k: calls.append("scores")
    )
    monkeypatch.setattr(
        plotting_module, "traces", lambda *a, **k: calls.append("traces")
    )
    monkeypatch.setattr(
        plotting_module, "thumbnail", lambda *a, **k: calls.append("thumbnail")
    )
    monkeypatch.setattr(
        plotting_module,
        "dendritic_feature_plots",
        lambda *a, **k: calls.append("dendritic"),
    )
    monkeypatch.setattr(
        plotting_module, "currentscape", lambda *a, **k: calls.append("currentscape")
    )
    monkeypatch.setattr(plotting_module, "FI_curve", lambda *a, **k: calls.append("fi"))
    monkeypatch.setattr(
        plotting_module, "plot_IV_curves", lambda *a, **k: calls.append("iv")
    )
    monkeypatch.setattr(
        plotting_module,
        "plot_FI_curves_comparison",
        lambda *a, **k: calls.append("fi_comparison"),
    )
    monkeypatch.setattr(
        plotting_module, "phase_plot", lambda *a, **k: calls.append("phase")
    )
    monkeypatch.setattr(
        plotting_module,
        "plot_trace_comparison",
        lambda *a, **k: calls.append("trace_comparison"),
    )
    monkeypatch.setattr(
        plotting_module, "run_and_plot_bAP", lambda *a, **k: calls.append("bap")
    )
    monkeypatch.setattr(
        plotting_module, "run_and_plot_EPSP", lambda *a, **k: calls.append("epsp")
    )
    monkeypatch.setattr(
        plotting_module,
        "run_and_plot_custom_sinespec",
        lambda *a, **k: calls.append("sinespec"),
    )

    result = plotting_module.plot_models(
        access_point,
        map,
        figures_dir=tmp_path,
        cell_evaluator=evaluator,
        plot_optimisation_progress=True,
        optimiser="IBEA",
        plot_parameter_evolution=True,
        plot_distributions=True,
        plot_scores=True,
        plot_traces=True,
        plot_thumbnail=True,
        plot_currentscape=True,
        plot_fi_curve=True,
        plot_dendritic_ISI_CV=True,
        plot_dendritic_rheobase=True,
        plot_bAP_EPSP=True,
        plot_IV_curve=True,
        plot_FI_curve_comparison=True,
        plot_phase_plot=True,
        plot_traces_comparison=True,
        run_plot_custom_sinspec=True,
        sinespec_settings=None,
    )

    assert result == [model]
    assert set(calls) == {
        "optimisation",
        "evolution",
        "distribution",
        "scores",
        "traces",
        "thumbnail",
        "dendritic",
        "currentscape",
        "fi",
        "iv",
        "fi_comparison",
        "phase",
        "trace_comparison",
        "bap",
        "epsp",
        "sinespec",
    }
