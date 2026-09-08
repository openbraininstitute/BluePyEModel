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

import logging
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import bluepyemodel.__main__ as package_main_module
from bluepyemodel.cli.analysis import _resolve_checkpoint_path
from bluepyemodel.cli.analysis import analyse
from bluepyemodel.cli.cli import main
from bluepyemodel.cli.optimisation import optimise


def test_main_help(cli_runner):
    result = cli_runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "BluePyEModel command-line interface." in result.output
    assert "--log-level" in result.output
    assert "optimise" in result.output
    assert "analyse" in result.output


def test_main_configures_log_level(cli_runner, recipes_path, analyse_mocks, monkeypatch):
    basic_config_calls = []
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )

    result = cli_runner.invoke(
        main,
        [
            "--log-level",
            "DEBUG",
            "analyse",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert basic_config_calls == [
        {
            "level": logging.DEBUG,
            "handlers": basic_config_calls[0]["handlers"],
            "force": True,
        }
    ]


def test_analyse_sets_fonttools_log_level(cli_runner, recipes_path, analyse_mocks):
    result = cli_runner.invoke(
        main,
        [
            "analyse",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert logging.getLogger("fontTools").level == logging.WARNING


def test_main_requires_subcommand(cli_runner):
    result = cli_runner.invoke(main, [])

    assert result.exit_code != 0


def test_main_unknown_subcommand(cli_runner):
    result = cli_runner.invoke(main, ["unknown"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_main_module_executes_main(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bluepyemodel", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("bluepyemodel.__main__", run_name="__main__")

    assert exc_info.value.code == 0


def test_main_module_runs_as_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "bluepyemodel", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--log-level" in result.stdout
    assert "optimise" in result.stdout
    assert "analyse" in result.stdout


def test_optimise_help(cli_runner):
    result = cli_runner.invoke(main, ["optimise", "--help"])

    assert result.exit_code == 0
    assert "--emodel" in result.output
    assert "--recipes-path" in result.output
    assert "--seed" in result.output
    assert "--workers" in result.output
    assert "--convert-checkpoint" in result.output
    assert "Run EModel optimisation." in result.output


def test_optimise_requires_emodel(cli_runner, recipes_path):
    result = cli_runner.invoke(
        main,
        ["optimise", "--seed", "42", "--recipes-path", str(recipes_path)],
    )

    assert result.exit_code != 0
    assert "Missing option '--emodel'" in result.output


def test_optimise_requires_recipes_path(cli_runner):
    result = cli_runner.invoke(main, ["optimise", "--seed", "42", "--emodel", "L5PC"])

    assert result.exit_code != 0
    assert "Missing option '--recipes-path'" in result.output


def test_optimise_requires_seed(cli_runner, recipes_path):
    result = cli_runner.invoke(
        main,
        ["optimise", "--emodel", "L5PC", "--recipes-path", str(recipes_path)],
    )

    assert result.exit_code != 0
    assert "Missing option '--seed'" in result.output


def test_optimise_rejects_invalid_seed(cli_runner, recipes_path):
    result = cli_runner.invoke(
        main,
        [
            "optimise",
            "--seed",
            "not-a-number",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--seed'" in result.output


def test_optimise_runs_optimisation(cli_runner, recipes_path, optimise_mocks):
    result = cli_runner.invoke(
        main,
        [
            "optimise",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
        ],
    )

    assert result.exit_code == 0, result.output
    optimise_mocks["local_access_point"].assert_called_once_with(
        emodel="L5PC",
        recipes_path=Path(recipes_path),
    )
    optimise_mocks["nested_pool_cls"].assert_called_once_with(processes=None)
    optimise_mocks["setup_and_run"].assert_called_once_with(
        optimise_mocks["access_point"],
        seed=7,
        mapper=optimise_mocks["nested_pool"].map,
        terminator=None,
        checkpoints_dir=Path("./checkpoints"),
    )
    optimise_mocks["pickle_to_hdf5"].assert_not_called()


def test_optimise_converts_checkpoint(cli_runner, recipes_path, optimise_mocks):
    result = cli_runner.invoke(
        main,
        [
            "optimise",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
            "--convert-checkpoint",
        ],
    )

    assert result.exit_code == 0, result.output
    optimise_mocks["get_checkpoint_path"].assert_called_once_with(
        optimise_mocks["emodel_metadata"],
        seed=7,
        base_dir=Path("./checkpoints"),
    )
    optimise_mocks["pickle_to_hdf5"].assert_called_once_with(
        Path("./checkpoints/emodel=L5PC__seed=7.pkl"),
        Path("./checkpoints/emodel=L5PC__seed=7.h5"),
    )


def test_optimise_uses_workers(cli_runner, recipes_path, optimise_mocks):
    result = cli_runner.invoke(
        main,
        [
            "optimise",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
            "--workers",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    optimise_mocks["nested_pool_cls"].assert_called_once_with(processes=4)


def test_optimise_command_direct(cli_runner, recipes_path, optimise_mocks):
    result = cli_runner.invoke(
        optimise,
        [
            "--seed",
            "3",
            "--emodel",
            "cADpyr_L5TPC",
            "--recipes-path",
            str(recipes_path),
        ],
    )

    assert result.exit_code == 0, result.output
    optimise_mocks["local_access_point"].assert_called_once_with(
        emodel="cADpyr_L5TPC",
        recipes_path=Path(recipes_path),
    )
    optimise_mocks["nested_pool_cls"].assert_called_once_with(processes=None)
    optimise_mocks["setup_and_run"].assert_called_once_with(
        optimise_mocks["access_point"],
        seed=3,
        mapper=optimise_mocks["nested_pool"].map,
        terminator=None,
        checkpoints_dir=Path("./checkpoints"),
    )


def test_analyse_help(cli_runner):
    result = cli_runner.invoke(main, ["analyse", "--help"])

    assert result.exit_code == 0
    assert "--emodel" in result.output
    assert "--recipes-path" in result.output
    assert "--seed" in result.output
    assert "--checkpoint-path" in result.output
    assert "Analyse an optimisation checkpoint" in result.output


def test_analyse_requires_emodel(cli_runner, recipes_path):
    result = cli_runner.invoke(
        main,
        ["analyse", "--seed", "42", "--recipes-path", str(recipes_path)],
    )

    assert result.exit_code != 0
    assert "Missing option '--emodel'" in result.output


def test_analyse_runs_analysis_pipeline(cli_runner, recipes_path, analyse_mocks):
    result = cli_runner.invoke(
        main,
        [
            "analyse",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
        ],
    )

    assert result.exit_code == 0, result.output
    analyse_mocks["local_access_point"].assert_called_once_with(
        emodel="L5PC",
        recipes_path=Path(recipes_path),
    )
    analyse_mocks["get_checkpoint_path"].assert_called_once()
    analyse_mocks["store_best_model"].assert_called_once_with(
        analyse_mocks["access_point"],
        seed=7,
        checkpoint_path=str(Path("./checkpoints/emodel=L5PC__seed=7.pkl")),
    )
    analyse_mocks["plot_models"].assert_called_once()
    plot_models_kwargs = analyse_mocks["plot_models"].call_args.kwargs
    assert plot_models_kwargs["access_point"] is analyse_mocks["access_point"]
    assert plot_models_kwargs["mapper"] is analyse_mocks["nested_pool"].map
    assert plot_models_kwargs["seeds"] == [7]
    assert plot_models_kwargs["only_validated"] is False
    analyse_mocks["export_emodels_sonata"].assert_called_once_with(
        analyse_mocks["access_point"],
        only_validated=False,
        only_best=False,
        seeds=[7],
        map_function=analyse_mocks["nested_pool"].map,
        output_dir=Path("./nodes"),
    )
    analyse_mocks["create_em_json"].assert_called_once_with(
        analyse_mocks["access_point"],
        seed=7,
        map_function=analyse_mocks["nested_pool"].map,
        output_dir=Path("./em"),
    )


def test_analyse_uses_explicit_checkpoint_path(cli_runner, recipes_path, analyse_mocks, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    result = cli_runner.invoke(
        main,
        [
            "analyse",
            "--seed",
            "3",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
            "--checkpoint-path",
            str(checkpoint_path),
        ],
    )

    assert result.exit_code == 0, result.output
    analyse_mocks["get_checkpoint_path"].assert_not_called()
    analyse_mocks["store_best_model"].assert_called_once_with(
        analyse_mocks["access_point"],
        seed=3,
        checkpoint_path=str(checkpoint_path),
    )


def test_analyse_command_direct(cli_runner, recipes_path, analyse_mocks):
    result = cli_runner.invoke(
        analyse,
        [
            "--seed",
            "2",
            "--emodel",
            "cADpyr_L5TPC",
            "--recipes-path",
            str(recipes_path),
        ],
    )

    assert result.exit_code == 0, result.output
    analyse_mocks["local_access_point"].assert_called_once_with(
        emodel="cADpyr_L5TPC",
        recipes_path=Path(recipes_path),
    )


def test_resolve_checkpoint_path_uses_explicit_path():
    access_point = MagicMock()

    resolved = _resolve_checkpoint_path(
        access_point,
        seed=7,
        checkpoint_path=Path("/tmp/checkpoint.h5"),
        checkpoints_dir=Path("./checkpoints"),
    )

    assert resolved == Path("/tmp/checkpoint.h5")


def test_resolve_checkpoint_path_uses_metadata_path(analyse_mocks):
    access_point = analyse_mocks["access_point"]
    checkpoints_dir = Path("/data/checkpoints")

    resolved = _resolve_checkpoint_path(
        access_point,
        seed=7,
        checkpoint_path=None,
        checkpoints_dir=checkpoints_dir,
    )

    analyse_mocks["get_checkpoint_path"].assert_called_once_with(
        access_point.emodel_metadata,
        seed=7,
        base_dir=checkpoints_dir,
    )
    assert resolved == Path("./checkpoints/emodel=L5PC__seed=7.pkl")


def _write_dummy_figures(**kwargs):
    figures_dir = kwargs["figures_dir"]
    nested_dir = figures_dir / "seed_7"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "trace.pdf").write_bytes(b"pdf")
    (nested_dir / "thumb.png").write_bytes(b"png")


def test_analyse_flattens_figure_outputs(cli_runner, recipes_path, analyse_mocks, tmp_path):
    analyse_mocks["plot_models"].side_effect = _write_dummy_figures
    output_figures_dir = tmp_path / "figures"

    result = cli_runner.invoke(
        main,
        [
            "analyse",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
            "--output-figures-dir",
            str(output_figures_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_figures_dir / "trace.pdf").is_file()
    assert (output_figures_dir / "thumb.png").is_file()
    assert not (output_figures_dir / "tmp").exists()


def test_analyse_uses_custom_output_dirs(cli_runner, recipes_path, analyse_mocks, tmp_path):
    output_figures_dir = tmp_path / "custom_figures"
    output_nodes_dir = tmp_path / "custom_nodes"
    output_em_dir = tmp_path / "custom_em"

    result = cli_runner.invoke(
        main,
        [
            "analyse",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
            "--output-figures-dir",
            str(output_figures_dir),
            "--output-nodes-dir",
            str(output_nodes_dir),
            "--output-em-dir",
            str(output_em_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    analyse_mocks["export_emodels_sonata"].assert_called_once_with(
        analyse_mocks["access_point"],
        only_validated=False,
        only_best=False,
        seeds=[7],
        map_function=analyse_mocks["nested_pool"].map,
        output_dir=output_nodes_dir,
    )
    analyse_mocks["create_em_json"].assert_called_once_with(
        analyse_mocks["access_point"],
        seed=7,
        map_function=analyse_mocks["nested_pool"].map,
        output_dir=output_em_dir,
    )


def test_optimise_uses_custom_checkpoints_dir(cli_runner, recipes_path, optimise_mocks, tmp_path):
    checkpoints_dir = tmp_path / "my_checkpoints"

    result = cli_runner.invoke(
        main,
        [
            "optimise",
            "--seed",
            "7",
            "--emodel",
            "L5PC",
            "--recipes-path",
            str(recipes_path),
            "--checkpoints-dir",
            str(checkpoints_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    optimise_mocks["setup_and_run"].assert_called_once_with(
        optimise_mocks["access_point"],
        seed=7,
        mapper=optimise_mocks["nested_pool"].map,
        terminator=None,
        checkpoints_dir=checkpoints_dir,
    )
