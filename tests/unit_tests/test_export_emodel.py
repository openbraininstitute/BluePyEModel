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

from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata


def test_export_emodels_sonata_passes_output_dir(tmp_path, monkeypatch):
    access_point = MagicMock()
    access_point.emodel_metadata.emodel = "L5PC"
    cell_evaluator = MagicMock()
    cell_model = MagicMock()
    cell_model.morphology.morph_modifiers = []
    cell_evaluator.cell_model = cell_model
    emodel = MagicMock()

    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel.get_evaluator_from_access_point",
        MagicMock(return_value=cell_evaluator),
    )
    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel.compute_responses",
        MagicMock(return_value=[emodel]),
    )
    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel.select_emodels",
        MagicMock(return_value=[emodel]),
    )
    export_model = MagicMock()
    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel._export_model_sonata",
        export_model,
    )

    output_dir = tmp_path / "sonata_out"
    export_emodels_sonata(
        access_point,
        only_validated=False,
        only_best=False,
        seeds=[7],
        map_function=map,
        output_dir=output_dir,
    )

    export_model.assert_called_once_with(
        cell_model,
        emodel,
        output_dir=output_dir,
        new_emodel_name=None,
    )


def test_export_emodels_sonata_stops_when_no_emodels_selected(monkeypatch):
    access_point = MagicMock()
    access_point.emodel_metadata.emodel = "L5PC"
    cell_evaluator = MagicMock()
    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel.get_evaluator_from_access_point",
        MagicMock(return_value=cell_evaluator),
    )
    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel.compute_responses",
        MagicMock(return_value=[MagicMock()]),
    )
    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel.select_emodels",
        MagicMock(return_value=[]),
    )
    export_model = MagicMock()
    monkeypatch.setattr(
        "bluepyemodel.export_emodel.export_emodel._export_model_sonata",
        export_model,
    )

    export_emodels_sonata(access_point, seeds=[1])

    export_model.assert_not_called()
