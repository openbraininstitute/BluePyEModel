"""
Copyright 2023-2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path

import pytest

from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata
from bluepyemodel.tools import search_pdfs
from tests.utils import cwd


@pytest.fixture
def metadata():
    return EModelMetadata(emodel="L5_TPC", etype="cAC", iteration_tag="v0")


def test_search_figure_path_no_match(tmp_path):
    with cwd(tmp_path):
        assert search_pdfs.search_figure_path("does_not_exist*.pdf") is None


def test_search_figure_path_single_match(tmp_path):
    with cwd(tmp_path):
        (tmp_path / "figure.pdf").touch()
        result = search_pdfs.search_figure_path("figure*.pdf")
        assert result == str((tmp_path / "figure.pdf").resolve())


def test_search_figure_path_raises_on_multiple_matches(tmp_path):
    with cwd(tmp_path):
        (tmp_path / "figure_a.pdf").touch()
        (tmp_path / "figure_b.pdf").touch()
        with pytest.raises(ValueError, match="More than one pdf"):
            search_pdfs.search_figure_path("figure_*.pdf")


def test_search_figure_paths_no_match(tmp_path):
    with cwd(tmp_path):
        assert search_pdfs.search_figure_paths("does_not_exist*.pdf") == []


def test_search_figure_paths_multiple_matches(tmp_path):
    with cwd(tmp_path):
        (tmp_path / "figure_a.pdf").touch()
        (tmp_path / "figure_b.pdf").touch()
        results = search_pdfs.search_figure_paths("figure_*.pdf")
        assert len(results) == 2


def test_figure_efeatures():
    pdf_amp, pdf_amp_rel = search_pdfs.figure_efeatures("L5_TPC", "IDrest_140", "voltage_base")
    assert pdf_amp == "./figures/L5_TPC/efeatures_extraction/*IDrest_140_voltage_base_amp.pdf"
    assert (
        pdf_amp_rel == "./figures/L5_TPC/efeatures_extraction/*IDrest_140_voltage_base_amp_rel.pdf"
    )


def test_search_figure_efeatures_no_match(tmp_path):
    with cwd(tmp_path):
        pdf_amp, pdf_amp_rel = search_pdfs.search_figure_efeatures(
            "L5_TPC", "IDrest_140", "voltage_base"
        )
        assert pdf_amp is None
        assert pdf_amp_rel is None


def test_figure_emodel_optimisation(metadata):
    path = search_pdfs.figure_emodel_optimisation(metadata, seed=1)
    assert path.name.endswith("__optimisation.pdf")
    assert path.parent.name == "optimisation"


def test_search_figure_emodel_optimisation_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        assert search_pdfs.search_figure_emodel_optimisation(metadata, seed=1) is None


def test_figure_emodel_traces(metadata):
    pathname, pathname_val = search_pdfs.figure_emodel_traces(metadata, seed=1)
    assert pathname.name.endswith("__traces.pdf")
    assert pathname.parent.name == "all"
    assert pathname_val.parent.name == "validated"


def test_search_figure_emodel_traces_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        results = search_pdfs.search_figure_emodel_traces(metadata, seed=1)
        assert results == [None, None]


def test_figure_emodel_score(metadata):
    pathname, pathname_val = search_pdfs.figure_emodel_score(metadata, seed=1)
    assert pathname.name.endswith("__scores.pdf")
    assert pathname.parent.name == "all"
    assert pathname_val.parent.name == "validated"


def test_search_figure_emodel_score_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        results = search_pdfs.search_figure_emodel_score(metadata, seed=1)
        assert results == [None, None]


def test_figure_emodel_thumbnail(metadata):
    pathname, pathname_val = search_pdfs.figure_emodel_thumbnail(metadata, seed=1)
    assert pathname.name.endswith("__thumbnail.png")
    assert pathname.parent.name == "all"
    assert pathname_val.parent.name == "validated"


def test_search_figure_emodel_thumbnail_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        results = search_pdfs.search_figure_emodel_thumbnail(metadata, seed=1)
        assert results == [None, None]


def test_figure_emodel_bAP(metadata):
    pathname, _ = search_pdfs.figure_emodel_bAP(metadata, seed=1)
    assert pathname.name.endswith("__dendrite_backpropagation_fit_decay.pdf")


def test_search_figure_emodel_bAP_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        results = search_pdfs.search_figure_emodel_bAP(metadata, seed=1)
        assert results == [None, None]


def test_figure_emodel_EPSP(metadata):
    pathname, _ = search_pdfs.figure_emodel_EPSP(metadata, seed=1)
    assert pathname.name.endswith("__dendrite_EPSP_attenuation_fit.pdf")


def test_search_figure_emodel_EPSP_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        results = search_pdfs.search_figure_emodel_EPSP(metadata, seed=1)
        assert results == [None, None]


def test_figure_emodel_ISI_CV(metadata):
    pathname, _ = search_pdfs.figure_emodel_ISI_CV(metadata, seed=1)
    assert "ISI_CV_linear.pdf" in pathname.name


def test_search_figure_emodel_ISI_CV_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        results = search_pdfs.search_figure_emodel_ISI_CV(metadata, seed=1)
        assert results == []


def test_figure_emodel_rheobase(metadata):
    pathname, _ = search_pdfs.figure_emodel_rheobase(metadata, seed=1)
    assert "bpo_threshold_current_linear.pdf" in pathname.name


def test_search_figure_emodel_rheobase_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        results = search_pdfs.search_figure_emodel_rheobase(metadata, seed=1)
        assert results == []


def test_figure_emodel_parameters(metadata):
    pathname, pathname_val = search_pdfs.figure_emodel_parameters(metadata)
    assert pathname.name.endswith("__parameters_distribution.pdf")
    assert pathname.parent.name == "all"
    assert pathname_val.parent.name == "validated"


def test_search_figure_emodel_parameters_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        results = search_pdfs.search_figure_emodel_parameters(metadata)
        assert results == [None, None]


def test_figure_emodel_parameters_evolution_with_seed(metadata):
    pathname = search_pdfs.figure_emodel_parameters_evolution(metadata, seed=1)
    assert pathname.name.endswith("__evo_parameter_density.pdf")


def test_figure_emodel_parameters_evolution_without_seed(metadata):
    pathname = search_pdfs.figure_emodel_parameters_evolution(metadata, seed=None)
    assert pathname.name.endswith("__all_seeds__evo_parameter_density.pdf")


def test_search_figure_emodel_parameters_evolution_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        assert search_pdfs.search_figure_emodel_parameters_evolution(metadata, seed=1) is None


def test_figure_emodel_currentscapes(metadata):
    pathname, pathname_val = search_pdfs.figure_emodel_currentscapes(metadata, seed=1)
    assert "currentscape" in pathname.name
    assert pathname.parent.name == "all"
    assert pathname_val.parent.name == "validated"


def test_search_figure_emodel_currentscapes_no_match(tmp_path, metadata):
    with cwd(tmp_path):
        assert search_pdfs.search_figure_emodel_currentscapes(metadata, seed=1) == []


def test_copy_emodel_pdf_dependency_to_new_path_copies_when_missing(tmp_path):
    old_path = tmp_path / "old" / "figure.pdf"
    old_path.parent.mkdir(parents=True)
    old_path.write_text("content")
    new_path = tmp_path / "new" / "figure.pdf"

    search_pdfs.copy_emodel_pdf_dependency_to_new_path(old_path, new_path)

    assert new_path.is_file()
    assert new_path.read_text() == "content"


def test_copy_emodel_pdf_dependency_to_new_path_noop_when_old_missing(tmp_path):
    old_path = tmp_path / "old" / "figure.pdf"
    new_path = tmp_path / "new" / "figure.pdf"

    search_pdfs.copy_emodel_pdf_dependency_to_new_path(old_path, new_path)

    assert not new_path.exists()


def test_copy_emodel_pdf_dependency_to_new_path_does_not_overwrite_by_default(tmp_path):
    old_path = tmp_path / "old" / "figure.pdf"
    old_path.parent.mkdir(parents=True)
    old_path.write_text("new content")
    new_path = tmp_path / "new" / "figure.pdf"
    new_path.parent.mkdir(parents=True)
    new_path.write_text("existing content")

    search_pdfs.copy_emodel_pdf_dependency_to_new_path(old_path, new_path, overwrite=False)

    assert new_path.read_text() == "existing content"


def test_copy_emodel_pdf_dependency_to_new_path_overwrites_when_asked(tmp_path):
    old_path = tmp_path / "old" / "figure.pdf"
    old_path.parent.mkdir(parents=True)
    old_path.write_text("new content")
    new_path = tmp_path / "new" / "figure.pdf"
    new_path.parent.mkdir(parents=True)
    new_path.write_text("existing content")

    search_pdfs.copy_emodel_pdf_dependency_to_new_path(old_path, new_path, overwrite=True)

    assert new_path.read_text() == "new content"


def test_copy_emodel_pdf_dependencies_to_new_path_copies_everything(tmp_path, metadata):
    new_metadata = EModelMetadata(emodel="L5_TPC", etype="cAC", iteration_tag="v1")

    with cwd(tmp_path):
        single_folder_fcts = [
            search_pdfs.figure_emodel_optimisation,
            search_pdfs.figure_emodel_parameters_evolution,
        ]
        two_folders_fcts = [
            search_pdfs.figure_emodel_traces,
            search_pdfs.figure_emodel_score,
            search_pdfs.figure_emodel_parameters,
            search_pdfs.figure_emodel_thumbnail,
        ]

        for fct in single_folder_fcts:
            old_path = fct(metadata, seed=1)
            old_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.write_text("data")

        for fct in two_folders_fcts:
            old_path, old_path_val = fct(metadata, seed=1)
            old_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.write_text("data")
            old_path_val.parent.mkdir(parents=True, exist_ok=True)
            old_path_val.write_text("data_val")

        all_evo_path = search_pdfs.figure_emodel_parameters_evolution(metadata, seed=None)
        all_evo_path.parent.mkdir(parents=True, exist_ok=True)
        all_evo_path.write_text("data_all_evo")

        currentscape_path, _ = search_pdfs.figure_emodel_currentscapes(
            metadata, seed=1
        )
        currentscape_path_real = Path(str(currentscape_path).replace("*", "IDrest_140"))
        currentscape_path_real.parent.mkdir(parents=True, exist_ok=True)
        currentscape_path_real.write_text("data_currentscape")

        search_pdfs.copy_emodel_pdf_dependencies_to_new_path(
            metadata, new_metadata, old_allen_notation=True, new_allen_notation=True, seed=1
        )

        for fct in single_folder_fcts:
            new_path = fct(new_metadata, seed=1)
            assert new_path.is_file()

        for fct in two_folders_fcts:
            new_path, new_path_val = fct(new_metadata, seed=1)
            assert new_path.is_file()
            assert new_path_val.is_file()

        new_all_evo_path = search_pdfs.figure_emodel_parameters_evolution(new_metadata, seed=None)
        assert new_all_evo_path.is_file()

        new_currentscape_path, _ = search_pdfs.figure_emodel_currentscapes(new_metadata, seed=1)
        new_currentscape_path_real = Path(str(new_currentscape_path).replace("*", "IDrest_140"))
        assert new_currentscape_path_real.is_file()
