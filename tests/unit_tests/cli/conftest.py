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

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from bluepyemodel.cli.cli import main
from bluepyemodel.cli.optimise import optimise
from tests.utils import DATA


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def recipes_path():
    return DATA / "config" / "recipes.json"


@pytest.fixture
def optimise_mocks(monkeypatch):
    access_point = object()
    local_access_point = MagicMock(return_value=access_point)

    nested_pool = MagicMock()
    nested_pool.map = MagicMock()
    nested_pool.__enter__ = MagicMock(return_value=nested_pool)
    nested_pool.__exit__ = MagicMock(return_value=False)
    nested_pool_cls = MagicMock(return_value=nested_pool)

    setup_and_run = MagicMock()

    monkeypatch.setattr("bluepyemodel.access_point.local.LocalAccessPoint", local_access_point)
    monkeypatch.setattr("bluepyemodel.tools.multiprocessing.NestedPool", nested_pool_cls)
    monkeypatch.setattr("bluepyemodel.optimisation.setup_and_run_optimisation", setup_and_run)

    return {
        "access_point": access_point,
        "local_access_point": local_access_point,
        "nested_pool": nested_pool,
        "nested_pool_cls": nested_pool_cls,
        "setup_and_run": setup_and_run,
    }
