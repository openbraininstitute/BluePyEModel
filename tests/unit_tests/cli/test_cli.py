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

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import bluepyemodel.__main__ as package_main_module
from bluepyemodel.cli.cli import main
from bluepyemodel.cli.optimise import optimise


def test_main_help(cli_runner):
    result = cli_runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "BluePyEModel command-line interface." in result.output
    assert "optimise" in result.output


def test_main_requires_subcommand(cli_runner):
    result = cli_runner.invoke(main, [])

    assert result.exit_code != 0


def test_main_unknown_subcommand(cli_runner):
    result = cli_runner.invoke(main, ["unknown"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_main_module_entry_point(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bluepyemodel", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(package_main_module.__file__), run_name="__main__")

    assert exc_info.value.code == 0


def test_main_module_runs_as_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "bluepyemodel", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "optimise" in result.stdout


def test_optimise_help(cli_runner):
    result = cli_runner.invoke(main, ["optimise", "--help"])

    assert result.exit_code == 0
    assert "--emodel" in result.output
    assert "--recipes-path" in result.output
    assert "--seed" in result.output
    assert "--workers" in result.output
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
    optimise_mocks["nested_pool_cls"].assert_called_once_with(processes="4")


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
    )
