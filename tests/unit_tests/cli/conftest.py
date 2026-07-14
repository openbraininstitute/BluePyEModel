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

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from bluepyemodel.cli.cli import main
from bluepyemodel.cli.optimisation import optimise
from tests.utils import DATA


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def recipes_path():
    return DATA / "config" / "recipes.json"


@pytest.fixture
def optimise_mocks(monkeypatch):
    access_point = MagicMock()
    emodel_metadata = MagicMock()
    access_point.emodel_metadata = emodel_metadata
    local_access_point = MagicMock(return_value=access_point)

    nested_pool = MagicMock()
    nested_pool.map = MagicMock()
    nested_pool.__enter__ = MagicMock(return_value=nested_pool)
    nested_pool.__exit__ = MagicMock(return_value=False)
    nested_pool_cls = MagicMock(return_value=nested_pool)

    setup_and_run = MagicMock()
    get_checkpoint_path = MagicMock(return_value="./checkpoints/emodel=L5PC__seed=7.pkl")
    pickle_to_hdf5 = MagicMock()

    monkeypatch.setattr("bluepyemodel.access_point.local.LocalAccessPoint", local_access_point)
    monkeypatch.setattr("bluepyemodel.tools.multiprocessing.NestedPool", nested_pool_cls)
    monkeypatch.setattr("bluepyemodel.optimisation.setup_and_run_optimisation", setup_and_run)
    monkeypatch.setattr("bluepyemodel.tools.utils.get_checkpoint_path", get_checkpoint_path)
    monkeypatch.setattr("bluepyemodel.tools.conversion.pickle_to_hdf5", pickle_to_hdf5)

    return {
        "access_point": access_point,
        "emodel_metadata": emodel_metadata,
        "local_access_point": local_access_point,
        "nested_pool": nested_pool,
        "nested_pool_cls": nested_pool_cls,
        "setup_and_run": setup_and_run,
        "get_checkpoint_path": get_checkpoint_path,
        "pickle_to_hdf5": pickle_to_hdf5,
    }


@pytest.fixture
def analyse_mocks(monkeypatch):
    access_point = MagicMock()
    emodel_metadata = MagicMock()
    access_point.emodel_metadata = emodel_metadata
    local_access_point = MagicMock(return_value=access_point)

    nested_pool = MagicMock()
    nested_pool.map = MagicMock()
    nested_pool.__enter__ = MagicMock(return_value=nested_pool)
    nested_pool.__exit__ = MagicMock(return_value=False)
    nested_pool_cls = MagicMock(return_value=nested_pool)

    pipeline = MagicMock()
    pipeline.access_point = access_point
    pipeline.mapper = nested_pool.map
    emodel_pipeline_cls = MagicMock(return_value=pipeline)

    store_best_model = MagicMock()
    export_emodels_sonata = MagicMock()
    create_em_json = MagicMock()
    get_checkpoint_path = MagicMock(return_value=Path("./checkpoints/emodel=L5PC__seed=7.pkl"))

    fake_emodel_pipeline_module = MagicMock()
    fake_emodel_pipeline_module.EModel_pipeline = emodel_pipeline_cls
    monkeypatch.setitem(
        sys.modules,
        "bluepyemodel.emodel_pipeline.emodel_pipeline",
        fake_emodel_pipeline_module,
    )
    monkeypatch.setattr("bluepyemodel.optimisation.store_best_model", store_best_model)
    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel.export_emodels_sonata",
        export_emodels_sonata,
    )
    monkeypatch.setattr("bluepyemodel.tools.create_em_json.create_em_json", create_em_json)
    monkeypatch.setattr("bluepyemodel.tools.utils.get_checkpoint_path", get_checkpoint_path)
    monkeypatch.setattr("bluepyemodel.tools.multiprocessing.NestedPool", nested_pool_cls)

    return {
        "access_point": access_point,
        "emodel_metadata": emodel_metadata,
        "pipeline": pipeline,
        "emodel_pipeline_cls": emodel_pipeline_cls,
        "nested_pool": nested_pool,
        "nested_pool_cls": nested_pool_cls,
        "store_best_model": store_best_model,
        "export_emodels_sonata": export_emodels_sonata,
        "create_em_json": create_em_json,
        "get_checkpoint_path": get_checkpoint_path,
    }


@pytest.fixture
def validate_mocks(monkeypatch):
    access_point = MagicMock()
    emodel_metadata = MagicMock()
    access_point.emodel_metadata = emodel_metadata

    nested_pool = MagicMock()
    nested_pool.map = MagicMock()
    nested_pool.__enter__ = MagicMock(return_value=nested_pool)
    nested_pool.__exit__ = MagicMock(return_value=False)
    nested_pool_cls = MagicMock(return_value=nested_pool)

    pipeline = MagicMock()
    pipeline.access_point = access_point
    pipeline.mapper = nested_pool.map
    emodel_pipeline_cls = MagicMock(return_value=pipeline)

    fake_emodel_pipeline_module = MagicMock()
    fake_emodel_pipeline_module.EModel_pipeline = emodel_pipeline_cls
    monkeypatch.setitem(
        sys.modules,
        "bluepyemodel.emodel_pipeline.emodel_pipeline",
        fake_emodel_pipeline_module,
    )
    monkeypatch.setattr("bluepyemodel.tools.multiprocessing.NestedPool", nested_pool_cls)

    return {
        "access_point": access_point,
        "pipeline": pipeline,
        "emodel_pipeline_cls": emodel_pipeline_cls,
        "nested_pool": nested_pool,
        "nested_pool_cls": nested_pool_cls,
    }
