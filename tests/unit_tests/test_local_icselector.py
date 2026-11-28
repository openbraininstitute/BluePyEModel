"""Configure a model from gene data with the local access point and the ion channel selector."""

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

import json
import shutil

import pytest

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.icselector import icselector
from bluepyemodel.model.model_configurator import ModelConfigurator
from tests.utils import DATA

EMODEL = "L5_TPC_A_cAC"
TTYPE = "0001 Test Ttype_1"
MTYPE = "L5_TPC:A"
ETYPE = "cAC"


def make_access_point(emodel_dir, pipeline_settings):
    recipes = {
        EMODEL: {
            "morph_path": "./morphology/",
            "morphology": [["L5TPCa", "C060114A5.asc"]],
            "params": "config/params.json",
            "features": "config/features.json",
            "pipeline_settings": pipeline_settings,
        }
    }
    recipes_path = emodel_dir / "config" / "recipes.json"
    recipes_path.write_text(json.dumps(recipes))
    return LocalAccessPoint(
        emodel=EMODEL,
        emodel_dir=emodel_dir,
        ttype=TTYPE,
        mtype=MTYPE,
        etype=ETYPE,
        recipes_path=recipes_path,
    )


@pytest.fixture
def emodel_dir(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copyfile(DATA / "config" / "params.json", tmp_path / "config" / "params.json")
    shutil.copytree(DATA / "icselector", tmp_path / "icselector")
    return tmp_path


@pytest.fixture
def access_point(emodel_dir):
    return make_access_point(
        emodel_dir,
        {
            "name_gene_map": "icselector/gene_map.csv",
            "path_ic_map": "icselector/ic_map.json",
        },
    )


def test_load_channel_gene_expression(access_point, emodel_dir):
    df, path = access_point.load_channel_gene_expression("icselector/gene_map.csv")

    assert path == emodel_dir / "icselector" / "gene_map.csv"
    assert list(df.index.names) == ["me-type", "t-type", "modality"]
    assert {"Kcna1", "Kcnc1"} <= set(df.columns)


def test_load_ic_map(access_point, emodel_dir):
    assert access_point.load_ic_map() == emodel_dir / "icselector" / "ic_map.json"


def test_absolute_paths_are_accepted(emodel_dir):
    access_point = make_access_point(
        emodel_dir, {"path_ic_map": str(emodel_dir / "icselector" / "ic_map.json")}
    )
    assert access_point.load_ic_map() == emodel_dir / "icselector" / "ic_map.json"


def test_get_t_types(access_point):
    # me-types are compared to the e-model name with non-alphanumeric characters replaced
    assert access_point.get_t_types("icselector/gene_map.csv") == [TTYPE]


def test_missing_setting_and_missing_file(emodel_dir):
    access_point = make_access_point(emodel_dir, {})
    with pytest.raises(ValueError, match="'path_ic_map' is required"):
        access_point.load_ic_map()
    with pytest.raises(ValueError, match="'name_gene_map' is required"):
        access_point.load_channel_gene_expression(None)

    access_point = make_access_point(emodel_dir, {"path_ic_map": "missing.json"})
    with pytest.raises(FileNotFoundError, match="missing.json"):
        access_point.load_ic_map()


def test_gene_based_configuration(access_point, monkeypatch):
    keys_passed = []
    original = icselector.ICSelector.get_cell_config_from_ttype

    def spy(self, key_words):
        keys_passed.append(key_words)
        return original(self, key_words)

    monkeypatch.setattr(icselector.ICSelector, "get_cell_config_from_ttype", spy)

    configurator = ModelConfigurator(access_point)
    configurator.new_configuration(use_gene_data=True)
    configuration = configurator.configuration

    # the selector receives the t-type and the me-type "<mtype>_<etype>"
    assert keys_passed == [[TTYPE, f"{MTYPE}_{ETYPE}"]]

    mechanisms = {(m.name, m.location): m.version for m in configuration.mechanisms}
    assert mechanisms[("Kv1_1", "somatic")] == "0122"
    assert mechanisms[("SKv3_1", "somatic")] == "sscx"
    assert ("pas", "somatic") in mechanisms

    parameter_names = {p.name for p in configuration.parameters}
    assert {"gkbar_Kv1_1", "gSKv3_1bar_SKv3_1", "celsius", "v_init"} <= parameter_names
