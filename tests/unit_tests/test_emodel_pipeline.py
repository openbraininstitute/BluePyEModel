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

import pytest

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.access_point import get_access_point
from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline_nexus
from tests.utils import DATA
from unittest.mock import patch


@pytest.fixture
def pipeline():

    pipe = EModel_pipeline(
        emodel="cADpyr_L5TPC",
        recipes_path=DATA / "config/recipes.json",
        ttype="test",
        species="mouse",
        brain_region="SSCX",
    )

    pipe.access_point.emodel_dir = DATA

    return pipe


def test_init_local(pipeline):
    assert isinstance(pipeline.access_point, LocalAccessPoint)


def test_nexus_access_point_was_removed():
    with pytest.raises(ValueError, match="removed in BluePyEModel 2.0"):
        EModel_pipeline(
            emodel="cADpyr_L5TPC",
            recipes_path=DATA / "config/recipes.json",
            data_access_point="nexus",
        )
    with pytest.raises(ValueError, match="removed in BluePyEModel 2.0"):
        get_access_point("nexus", emodel="cADpyr_L5TPC")
    with pytest.raises(ValueError, match="removed in BluePyEModel 2.0"):
        EModel_pipeline_nexus(emodel="cADpyr_L5TPC")


def test_unknown_access_point():
    with pytest.raises(ValueError, match="Unknown access point"):
        EModel_pipeline(emodel="cADpyr_L5TPC", data_access_point="other")
    with pytest.raises(ValueError, match="Unknown access point"):
        get_access_point("other", emodel="cADpyr_L5TPC")


def test_nexus_arguments_are_ignored_with_a_warning():
    with pytest.warns(DeprecationWarning, match="nexus_project, access_token ignored"):
        pipe = EModel_pipeline(
            emodel="cADpyr_L5TPC",
            recipes_path=DATA / "config/recipes.json",
            nexus_project="project",
            access_token="token",
        )
    assert isinstance(pipe.access_point, LocalAccessPoint)
