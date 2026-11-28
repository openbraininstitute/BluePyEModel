"""Tests of the CLI subcommands for the pipeline steps, with a mocked pipeline."""

"""
Copyright 2025, Open Brain Institute

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

import logging
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from bluepyemodel.cli import common
from bluepyemodel.cli.cli import main

BASE = ["--emodel", "L5PC", "--recipes-path", "recipes.json", "--backend", "serial"]


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """Replace EModel_pipeline by a mock, and record how it was created."""
    pipeline = MagicMock()
    pipeline.access_point.emodel_dir = tmp_path
    pipeline.access_point.get_recipes.return_value = {
        "features": "features.json",
        "params": "params.json",
    }
    created = MagicMock(return_value=pipeline)
    monkeypatch.setattr("bluepyemodel.emodel_pipeline.emodel_pipeline.EModel_pipeline", created)
    pipeline.created = created
    return pipeline


@pytest.fixture
def exports(monkeypatch):
    hoc, sonata = MagicMock(), MagicMock()
    monkeypatch.setattr("bluepyemodel.export_emodel.export_emodel.export_emodels_hoc", hoc)
    monkeypatch.setattr("bluepyemodel.export_emodel.export_emodel.export_emodels_sonata", sonata)
    return hoc, sonata


def invoke(*args):
    result = CliRunner().invoke(main, list(args), catch_exceptions=False)
    return result


@pytest.mark.parametrize(
    "command",
    ["extract", "store", "validate", "plot", "summarize", "export", "run"],
)
def test_commands_are_registered(command):
    result = invoke(command, "--help")
    assert result.exit_code == 0
    assert "--emodel" in result.output
    assert "--brain-region" in result.output


def test_metadata_options_are_passed_to_the_pipeline(pipeline):
    result = invoke(
        "validate", *BASE, "--etype", "cADpyr", "--species", "rat", "--brain-region", "SSCX"
    )

    assert result.exit_code == 0, result.output
    pipeline.created.assert_called_once_with(
        emodel="L5PC",
        recipes_path="recipes.json",
        data_access_point="local",
        etype="cADpyr",
        species="rat",
        brain_region="SSCX",
    )
    pipeline.validation.assert_called_once_with()


def test_backend_sets_the_pipeline_mapper(pipeline):
    invoke("validate", *BASE)
    assert pipeline.mapper is map


def test_extract(pipeline, tmp_path):
    pipeline.extract_efeatures.return_value.efeatures = [1, 2, 3]
    pipeline.extract_efeatures.return_value.protocols = [1]

    result = invoke("extract", *BASE)

    assert result.exit_code == 0, result.output
    assert "Extracted 3 e-features for 1 protocols" in result.output
    assert str(tmp_path / "features.json") in result.output


def test_configure_model(pipeline):
    pipeline.configure_model.return_value.mechanisms = [1, 2]
    pipeline.configure_model.return_value.parameters = [1, 2, 3]

    result = invoke("configure-model", *BASE, "--morphology-name", "C1", "--use-gene-data")

    assert result.exit_code == 0, result.output
    pipeline.configure_model.assert_called_once_with(
        morphology_name="C1", morphology_path=None, morphology_format=None, use_gene_data=True
    )
    assert "2 mechanisms and 3 parameters" in result.output


def test_store_all_seeds_or_given_seeds(pipeline):
    invoke("store", *BASE)
    pipeline.store_optimisation_results.assert_called_once_with()

    pipeline.store_optimisation_results.reset_mock()
    invoke("store", *BASE, "--seed", "1", "--seed", "3")
    assert [c.kwargs for c in pipeline.store_optimisation_results.call_args_list] == [
        {"seed": 1},
        {"seed": 3},
    ]


def test_plot(pipeline):
    invoke("plot", *BASE, "--seed", "2", "--only-validated")
    pipeline.plot.assert_called_once_with(only_validated=True, seeds=[2])


@pytest.mark.parametrize("export_format", ["hoc", "sonata"])
def test_export(pipeline, exports, tmp_path, export_format):
    hoc, sonata = exports
    result = invoke(
        "export", *BASE, "--format", export_format, "--seed", "1", "--output-dir", str(tmp_path)
    )

    assert result.exit_code == 0, result.output
    called, not_called = (hoc, sonata) if export_format == "hoc" else (sonata, hoc)
    called.assert_called_once()
    not_called.assert_not_called()
    kwargs = called.call_args.kwargs
    assert kwargs["seeds"] == [1]
    assert kwargs["output_dir"] == tmp_path
    assert kwargs["only_best"] is True


def test_export_output_dir_needs_one_seed(pipeline, exports):
    result = CliRunner().invoke(main, ["export", *BASE, "--output-dir", "out"])
    assert result.exit_code != 0
    assert "--output-dir needs exactly one --seed" in result.output


def test_run_extracts_when_the_features_file_is_missing(pipeline, exports):
    result = invoke("run", *BASE, "--seed", "1", "--seed", "2", "--no-plot")

    assert result.exit_code == 0, result.output
    pipeline.extract_efeatures.assert_called_once_with()
    assert [c.kwargs for c in pipeline.optimise.call_args_list] == [{"seed": 1}, {"seed": 2}]
    pipeline.validation.assert_called_once_with()
    pipeline.plot.assert_not_called()
    exports[1].assert_called_once()


def test_run_skips_extraction_when_the_features_file_exists(pipeline, exports, tmp_path):
    (tmp_path / "features.json").write_text("{}")

    invoke("run", *BASE)
    pipeline.extract_efeatures.assert_not_called()

    invoke("run", *BASE, "--force-extraction")
    pipeline.extract_efeatures.assert_called_once_with()


def test_given_metadata_drops_missing_values():
    assert common.given_metadata(etype="a", species=None) == {"etype": "a"}


def test_run_exports_to_hoc(pipeline, exports, tmp_path):
    (tmp_path / "features.json").write_text("{}")
    invoke("run", *BASE, "--export-format", "hoc", "--no-plot")
    hoc, sonata = exports
    hoc.assert_called_once()
    sonata.assert_not_called()


def test_mapper_multiprocessing(monkeypatch):
    pool = MagicMock()
    pool.__enter__.return_value = pool
    pool_cls = MagicMock(return_value=pool)
    monkeypatch.setattr("bluepyemodel.tools.multiprocessing.NestedPool", pool_cls)

    with common.mapper("multiprocessing", workers=3) as map_function:
        assert map_function is pool.map

    pool_cls.assert_called_once_with(processes=3)
    pool.__exit__.assert_called_once()


def test_mapper_ipyparallel(monkeypatch):
    ipyparallel_map = MagicMock()
    factory = MagicMock(return_value=ipyparallel_map)
    monkeypatch.setattr("bluepyemodel.tools.multiprocessing.ipyparallel_map_function", factory)

    with common.mapper("ipyparallel", ipyparallel_profile="profile") as map_function:
        assert map_function is ipyparallel_map

    factory.assert_called_once_with(profile="profile")


def test_mapper_serial():
    with common.mapper("serial") as map_function:
        assert map_function is map


def test_summarize(pipeline):
    pipeline.access_point.__str__ = MagicMock(return_value="summary of L5PC")
    result = invoke("summarize", *BASE)
    assert "summary of L5PC" in result.output


def test_fonttools_logs_are_quiet(pipeline):
    invoke("--log-level", "DEBUG", "validate", *BASE)
    assert logging.getLogger("fontTools").level == logging.WARNING
