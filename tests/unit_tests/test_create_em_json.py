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

import json
from unittest.mock import MagicMock

import pytest

from bluepyemodel.tools.create_em_json import create_em_json


def test_create_em_json_writes_file(tmp_path, monkeypatch):
    access_point = MagicMock()
    access_point.emodel_metadata.as_string.return_value = "emodel=L5PC__seed=7"
    cell_evaluator = MagicMock()
    emodel = MagicMock()
    emodel.as_dict.return_value = {"name": "L5PC", "seed": 7}

    monkeypatch.setattr(
        "bluepyemodel.tools.create_em_json.get_evaluator_from_access_point",
        MagicMock(return_value=cell_evaluator),
    )
    monkeypatch.setattr(
        "bluepyemodel.tools.create_em_json.compute_responses",
        MagicMock(return_value=[emodel]),
    )

    output_path = create_em_json(
        access_point,
        seed=7,
        map_function=map,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "EM__emodel=L5PC__seed=7.json"
    with output_path.open() as json_file:
        payload = json.load(json_file)
    assert payload == {"name": "L5PC", "seed": 7}


def test_create_em_json_raises_when_no_emodel(monkeypatch):
    access_point = MagicMock()
    monkeypatch.setattr(
        "bluepyemodel.tools.create_em_json.get_evaluator_from_access_point",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "bluepyemodel.tools.create_em_json.compute_responses",
        MagicMock(return_value=[]),
    )

    with pytest.raises(ValueError, match="No emodel found for seed 9"):
        create_em_json(access_point, seed=9)
