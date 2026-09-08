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
from unittest.mock import MagicMock

from bluepyemodel.optimisation.optimisation import setup_and_run_optimisation


def test_setup_and_run_optimisation_uses_checkpoints_dir(tmp_path, monkeypatch):
    access_point = MagicMock()
    access_point.pipeline_settings.use_stagnation_criterion = True
    access_point.pipeline_settings.optimisation_params = {"seed": 1}
    access_point.pipeline_settings.optimiser = "IBEA"
    access_point.pipeline_settings.optimisation_checkpoint_period = None
    access_point.pipeline_settings.max_ngen = 2
    access_point.emodel_metadata = MagicMock()

    cell_evaluator = MagicMock()
    cell_evaluator.param_names = ["gNa"]
    monkeypatch.setattr(
        "bluepyemodel.optimisation.optimisation.get_evaluator_from_access_point",
        MagicMock(return_value=cell_evaluator),
    )
    optimiser = MagicMock()
    monkeypatch.setattr(
        "bluepyemodel.optimisation.optimisation.setup_optimiser",
        MagicMock(return_value=optimiser),
    )
    run_optimisation = MagicMock()
    monkeypatch.setattr(
        "bluepyemodel.optimisation.optimisation.run_optimisation",
        run_optimisation,
    )
    get_checkpoint_path = MagicMock(return_value="custom/checkpoint.pkl")
    monkeypatch.setattr(
        "bluepyemodel.optimisation.optimisation.get_checkpoint_path",
        get_checkpoint_path,
    )

    checkpoints_dir = tmp_path / "custom_checkpoints"
    setup_and_run_optimisation(
        access_point,
        seed=7,
        mapper=map,
        checkpoints_dir=checkpoints_dir,
    )

    get_checkpoint_path.assert_called_once_with(
        access_point.emodel_metadata,
        7,
        base_dir=checkpoints_dir,
    )
    run_optimisation.assert_called_once_with(
        optimiser=optimiser,
        checkpoint_path="custom/checkpoint.pkl",
        max_ngen=2,
        terminator=None,
        optimisation_checkpoint_period=None,
    )
