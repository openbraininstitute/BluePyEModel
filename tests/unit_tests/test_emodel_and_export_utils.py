"""Tests for deterministic EModel serialization and export helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy
import pytest

from bluepyemodel.emodel_pipeline.emodel import EModel, format_dict_for_resource
from bluepyemodel.export_emodel.utils import (
    copy_hocs_to_new_output_path,
    get_hoc_file_path,
    get_output_path,
    get_output_path_from_metadata,
    select_emodels,
)


def test_format_dict_for_resource_handles_none_and_nan():
    assert format_dict_for_resource(None) == []
    assert format_dict_for_resource(
        {"fixed": 1.5, "missing": None, "nan": numpy.nan}
    ) == [
        {"name": "fixed", "value": 1.5, "unitCode": ""},
        {"name": "missing", "value": None, "unitCode": ""},
        {"name": "nan", "value": None, "unitCode": ""},
    ]


def test_build_pdf_dependencies_filters_lists_and_keeps_scalar_results(monkeypatch):
    metadata = object()
    results = {
        "search_figure_emodel_optimisation": "optimisation.pdf",
        "search_figure_emodel_traces": ["traces.pdf", None, ""],
        "search_figure_emodel_score": ["score.pdf"],
        "search_figure_emodel_thumbnail": [],
        "search_figure_emodel_parameters": [None, "parameters.pdf"],
        "search_figure_emodel_parameters_evolution": "seed-evolution.pdf",
        "search_figure_emodel_currentscapes": ["currentscape.pdf", None],
        "search_figure_emodel_bAP": ["bap.pdf"],
        "search_figure_emodel_EPSP": None,
        "search_figure_emodel_ISI_CV": ["isi-cv.pdf"],
        "search_figure_emodel_rheobase": [],
    }

    def evolution(_metadata, seed):
        return (
            results["search_figure_emodel_parameters_evolution"] if seed else "all.pdf"
        )

    for name, result in results.items():
        if name == "search_figure_emodel_parameters_evolution":
            monkeypatch.setattr(
                "bluepyemodel.emodel_pipeline.emodel.search_pdfs." + name,
                evolution,
            )
        else:
            monkeypatch.setattr(
                "bluepyemodel.emodel_pipeline.emodel.search_pdfs." + name,
                MagicMock(return_value=result),
            )

    model = EModel(emodel_metadata=metadata, seed=7)

    assert model.build_pdf_dependencies(seed=7) == [
        "optimisation.pdf",
        "traces.pdf",
        "score.pdf",
        "parameters.pdf",
        "seed-evolution.pdf",
        "all.pdf",
        "currentscape.pdf",
        "bap.pdf",
        "isi-cv.pdf",
    ]


def test_emodel_converts_resources_and_serializes(monkeypatch):
    model = EModel(
        parameter=[{"name": "gNa", "value": 0.1}],
        score={"spike": 1.25, "width": 2.75},
        features=[{"name": "missing_value"}],
        scoreValidation=[{"name": "validation", "value": numpy.nan}],
        passedValidation=True,
        seed=3,
        workflow_id="workflow-id",
        threshold_data={"rmp": -70},
    )
    monkeypatch.setattr(model, "build_pdf_dependencies", lambda seed: [f"pdf-{seed}"])

    assert model.parameters == {"gNa": 0.1}
    assert numpy.isnan(model.features["missing_value"])
    assert model.get_related_nexus_ids()["generation"]["activity"]["followedWorkflow"][
        "id"
    ] == ("workflow-id")
    assert model.as_dict() == {
        "fitness": 4.0,
        "parameter": [{"name": "gNa", "value": 0.1, "unitCode": ""}],
        "score": [
            {"name": "spike", "value": 1.25, "unitCode": ""},
            {"name": "width", "value": 2.75, "unitCode": ""},
        ],
        "features": [{"name": "missing_value", "value": None, "unitCode": ""}],
        "scoreValidation": [{"name": "validation", "value": None, "unitCode": ""}],
        "passedValidation": True,
        "nexus_images": ["pdf-3"],
        "seed": 3,
        "threshold_data": {"rmp": -70},
    }


def test_copy_pdf_dependencies_delegates_to_search_helper(monkeypatch):
    model = EModel(emodel_metadata="metadata")
    copy = MagicMock()
    monkeypatch.setattr(
        "bluepyemodel.emodel_pipeline.emodel.search_pdfs.copy_emodel_pdf_dependencies_to_new_path",
        copy,
    )

    model.copy_pdf_dependencies_to_new_path(seed=11, overwrite=True)

    copy.assert_called_once_with(
        "metadata", "metadata", False, True, 11, overwrite=True
    )


def test_output_paths_and_hoc_path(tmp_path):
    metadata = MagicMock()
    metadata.as_string.return_value = "emodel=test__seed=4"
    emodel = SimpleNamespace(emodel_metadata=metadata, seed=4)

    assert get_output_path_from_metadata("exports", metadata, 4) == (
        "./exports/emodel=test__seed=4/"
    )
    output_path = get_output_path(emodel, output_dir=tmp_path / "explicit")
    assert output_path.is_dir()
    assert get_hoc_file_path(output_path) == str(output_path / "model.hoc")

    metadata.as_string.assert_called_once_with(seed=4, use_allen_notation=True)


def test_copy_hocs_to_new_output_path_copies_only_when_needed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Metadata:
        def as_string(self, seed, use_allen_notation):
            return "allen" if use_allen_notation else "legacy"

    emodel = SimpleNamespace(emodel_metadata=Metadata(), seed=1)
    old_path = tmp_path / "exports" / "legacy"
    old_path.mkdir(parents=True)
    (old_path / "model.hoc").write_text("hoc")
    (old_path / "extra.mod").write_text("mod")

    copy_hocs_to_new_output_path(emodel, "exports")

    new_path = tmp_path / "exports" / "allen"
    assert (new_path / "model.hoc").read_text() == "hoc"
    assert (new_path / "extra.mod").read_text() == "mod"

    (new_path / "model.hoc").write_text("updated")
    copy_hocs_to_new_output_path(emodel, "exports")
    assert (new_path / "model.hoc").read_text() == "updated"


@pytest.fixture
def emodels():
    metadata = lambda iteration: SimpleNamespace(iteration=iteration)
    return [
        SimpleNamespace(
            emodel_metadata=metadata("first"),
            fitness=3.0,
            seed=1,
            passed_validation=False,
        ),
        SimpleNamespace(
            emodel_metadata=metadata("second"),
            fitness=1.0,
            seed=2,
            passed_validation=True,
        ),
    ]


def test_select_emodels_filters_iteration_and_validation(emodels):
    assert select_emodels("model", emodels, only_best=False, iteration="second") == [
        emodels[1]
    ]
    assert select_emodels("model", emodels, only_best=False, only_validated=True) == [
        emodels[1]
    ]
    assert select_emodels("model", [], only_best=False) == []


def test_select_emodels_selects_best_and_filters_seed(emodels):
    assert select_emodels("model", emodels) == [emodels[1]]
    assert select_emodels("model", emodels, only_best=False, seeds=[1]) == [emodels[0]]
    assert select_emodels("model", emodels, only_best=False, seeds=[99]) == []
    assert (
        select_emodels(
            "model", emodels, only_best=False, only_validated=True, seeds=[1]
        )
        == []
    )
