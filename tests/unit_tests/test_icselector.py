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

import pytest

from bluepyemodel.icselector.icselector import ICSelector
from bluepyemodel.icselector.modules.configuration import Configuration
from bluepyemodel.icselector.modules.distribution import Distribution
from bluepyemodel.icselector.modules.mechanism import Mechanism
from bluepyemodel.icselector.modules.model_selector import ModelSelector
from tests.utils import DATA

ICSELECTOR_DATA = DATA / "icselector"


@pytest.fixture
def ic_map():
    with open(ICSELECTOR_DATA / "ic_map.json", encoding="utf-8") as fid:
        return json.load(fid)


@pytest.fixture
def icselector():
    return ICSelector(
        ic_map_path=ICSELECTOR_DATA / "ic_map.json",
        gene_map_path=ICSELECTOR_DATA / "gene_map.csv",
    )


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


def test_distribution_defaults():
    distr = Distribution()
    assert distr.somatic == "uniform"
    assert distr.all == ""


def test_distribution_set_all():
    distr = Distribution()
    distr.set_all("exp")
    assert distr.somatic == "exp"
    assert distr.basal == "exp"
    assert distr.apical == "exp"
    assert distr.axonal == "exp"
    # 'all' field itself is untouched by set_all
    assert distr.all == ""


def test_distribution_set_fields_with_all():
    distr = Distribution()
    distr.set_fields(all="exp")
    assert distr.all == "exp"
    # other fields are emptied when 'all' is set
    assert distr.somatic == ""
    assert distr.basal == ""


def test_distribution_set_fields_nan_becomes_empty():
    distr = Distribution()
    distr.set_fields(somatic="nan", axonal="exp")
    assert distr.somatic == ""
    assert distr.axonal == "exp"


def test_distribution_set_fields_non_string_becomes_empty():
    distr = Distribution()
    distr.set_fields(somatic=1.0)
    assert distr.somatic == ""


def test_distribution_get_excludes_empty():
    distr = Distribution()
    distr.set_fields(somatic="uniform", axonal="")
    result = distr.get()
    assert "somatic" in result
    assert "axonal" not in result


def test_distribution_str():
    distr = Distribution()
    distr.set_fields(somatic="uniform")
    assert "somatic" in str(distr)


# ---------------------------------------------------------------------------
# Mechanism
# ---------------------------------------------------------------------------


@pytest.fixture
def pas_mechanism():
    return Mechanism(model={"suffix": "pas", "gbar": "g", "parameters": ["e"]}, status="stable")


def test_mechanism_defaults(pas_mechanism):
    assert pas_mechanism.is_selected() is False
    assert pas_mechanism._bounds["g"] == [0, 1]
    assert pas_mechanism._bounds["e"] == [0]


def test_mechanism_select_no_status():
    mech = Mechanism(model={"suffix": "test"})
    mech.select()
    assert mech.is_selected() is True


def test_mechanism_select_matching_status(pas_mechanism):
    pas_mechanism.select(check_status="stable")
    assert pas_mechanism.is_selected() is True


def test_mechanism_select_only_stable_is_strict(pas_mechanism):
    # Only check_status == "stable" is checked strictly against mech.status;
    # any other value (e.g. "latest") selects unconditionally.
    pas_mechanism.select(check_status="latest")
    assert pas_mechanism.is_selected() is True


def test_mechanism_select_stable_mismatch(pas_mechanism):
    pas_mechanism.status = "latest"
    pas_mechanism.select(check_status="stable")
    assert pas_mechanism.is_selected() is False


def test_mechanism_deselect(pas_mechanism):
    pas_mechanism.select()
    pas_mechanism.deselect()
    assert pas_mechanism.is_selected() is False


def test_mechanism_get_bounds(pas_mechanism):
    assert pas_mechanism.get_bounds("g") == [0, 1]


def test_mechanism_set_gbar_list(pas_mechanism):
    pas_mechanism.set_gbar([1e-5, 6e-5])
    assert pas_mechanism.get_bounds("g") == [1e-5, 6e-5]


def test_mechanism_set_gbar_scalar(pas_mechanism):
    pas_mechanism.set_gbar(5e-5)
    assert pas_mechanism.get_bounds("g") == [5e-5]


def test_mechanism_set_gbar_raises_on_non_numeric(pas_mechanism):
    with pytest.raises(TypeError):
        pas_mechanism.set_gbar(["a"])


def test_mechanism_set_parameters(pas_mechanism):
    pas_mechanism.set_parameters(e=[-95, -60])
    assert pas_mechanism.get_bounds("e") == [-95, -60]


def test_mechanism_set_parameters_scalar(pas_mechanism):
    pas_mechanism.set_parameters(e=-70)
    assert pas_mechanism.get_bounds("e") == [-70]


def test_mechanism_set_parameters_raises_on_unknown_param(pas_mechanism):
    with pytest.raises(KeyError):
        pas_mechanism.set_parameters(unknown_param=1)


def test_mechanism_set_distribution_from_str(pas_mechanism):
    pas_mechanism.set_distribution("exp")
    assert pas_mechanism.distribution.somatic == "exp"


def test_mechanism_set_distribution_from_kwargs(pas_mechanism):
    pas_mechanism.set_distribution(somatic="exp")
    assert pas_mechanism.distribution.somatic == "exp"


def test_mechanism_set_distribution_from_distribution_instance(pas_mechanism):
    distr = Distribution()
    distr.set_fields(somatic="exp")
    pas_mechanism.set_distribution(distr)
    assert pas_mechanism.distribution.somatic == "exp"


def test_mechanism_set_from_gene_info(pas_mechanism):
    distr = Distribution()
    distr.set_fields(somatic="exp")
    info = {"channel": "K_Tst", "distribution": distr, "gbar_max": 0.5}
    pas_mechanism.set_from_gene_info(info)
    assert pas_mechanism.get_bounds("g") == [0, 0.5]
    assert pas_mechanism._mapped_from == ["K_Tst"]


def test_mechanism_set_from_icmap_list():
    mech = Mechanism(model={"suffix": "K_Tst", "gbar": "gK_Tstbar"})
    mech.set_from_icmap([0, 1])
    assert mech.get_bounds("gK_Tstbar") == [0, 1]


def test_mechanism_set_from_icmap_dict(pas_mechanism):
    pas_mechanism.set_from_icmap(
        {"gbar": [1e-5, 6e-5], "distribution": "uniform", "bounds": {"e": [-95, -60]}}
    )
    assert pas_mechanism.get_bounds("g") == [1e-5, 6e-5]
    assert pas_mechanism.get_bounds("e") == [-95, -60]
    assert pas_mechanism.distribution.somatic == "uniform"


def test_mechanism_set_from_icmap_dict_distribution_kwargs():
    mech = Mechanism(model={"suffix": "test", "gbar": "g"})
    mech.set_from_icmap({"distribution": {"somatic": "exp"}})
    assert mech.distribution.somatic == "exp"


def test_mechanism_asdict(pas_mechanism):
    d = pas_mechanism.asdict()
    assert d["model"]["suffix"] == "pas"


def test_mechanism_str(pas_mechanism):
    pas_mechanism.set_parameters(e=[-95, -60])
    out = str(pas_mechanism)
    assert "pas" in out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_configuration_add_from_mechanism():
    mech = Mechanism(model={"suffix": "K_Tst", "gbar": "gK_Tstbar", "parameters": []})
    mech.set_distribution("uniform")
    mech.set_gbar([0, 1])

    config = Configuration()
    config.add_from_mechanism(mech)

    params = config.get_parameters()
    assert len(params) > 0
    assert any(p["mechanism"] == "K_Tst" for p in params)


def test_configuration_add_parameter():
    config = Configuration()
    config.add_parameter("v_init", "global", (-80, -80))
    params = config.get_parameters()
    assert len(params) == 1
    assert params[0]["name"] == "v_init"


def test_configuration_get_mechanisms():
    mech = Mechanism(model={"suffix": "K_Tst", "gbar": "gK_Tstbar", "parameters": []})
    mech.set_distribution("uniform")
    mech.set_gbar([0, 1])

    config = Configuration()
    config.add_from_mechanism(mech)

    mechs = config.get_mechanisms()
    assert {"name": "K_Tst", "location": "somatic"} in mechs


def test_configuration_get_distributions():
    mech = Mechanism(model={"suffix": "K_Tst", "gbar": "gK_Tstbar", "parameters": []})
    mech.set_distribution("exp")
    mech.set_gbar([0, 1])

    config = Configuration()
    config.add_from_mechanism(mech)

    distrs = config.get_distributions()
    assert {"name": "exp"} in distrs


def test_configuration_str():
    config = Configuration()
    config.add_parameter("v_init", "global", (-80, -80))
    out = str(config)
    assert "global" in out
    assert "v_init" in out


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------


def test_model_selector_get_by_model_name(ic_map):
    selector = ModelSelector(ic_map)
    mech = selector.get("K_Tst")
    assert mech is not None
    assert mech.model["suffix"] == "K_Tst"


def test_model_selector_get_by_channel_name(ic_map):
    selector = ModelSelector(ic_map)
    selector.mode = "genetic"
    mech = selector.get("Kv1.1")
    assert mech is not None
    assert mech.model["suffix"] == "Kv1_1"


def test_model_selector_get_unknown_returns_none(ic_map):
    selector = ModelSelector(ic_map)
    assert selector.get("does_not_exist") is None


def test_model_selector_select(ic_map):
    selector = ModelSelector(ic_map)
    mech = selector.select("pas")
    assert mech.is_selected() is True


def test_model_selector_select_unknown_returns_none(ic_map):
    selector = ModelSelector(ic_map)
    assert selector.select("does_not_exist") is None


def test_model_selector_get_mechanisms_selected_only(ic_map):
    selector = ModelSelector(ic_map)
    selector.select("pas")
    mechs = selector.get_mechanisms(selected_only=True)
    assert "pas" in mechs
    assert all(m.is_selected() for m in mechs.values())


def test_model_selector_get_mechanisms_all(ic_map):
    selector = ModelSelector(ic_map)
    mechs = selector.get_mechanisms(selected_only=False)
    assert "pas" in mechs
    assert "K_Tst" in mechs


def test_model_selector_str(ic_map):
    selector = ModelSelector(ic_map)
    selector.select("pas")
    assert "Mechanisms" in str(selector)


# ---------------------------------------------------------------------------
# ICSelector (integration of the whole module)
# ---------------------------------------------------------------------------


def test_icselector_init(icselector):
    assert icselector._model_selector.status == "latest"
    assert icselector._model_selector.mode == "mixed"


def test_icselector_set_status(icselector):
    icselector.set_status("stable")
    assert icselector._model_selector.status == "stable"
    # invalid values are ignored
    icselector.set_status("invalid")
    assert icselector._model_selector.status == "stable"


def test_icselector_set_mode(icselector):
    icselector.set_mode("genetic")
    assert icselector._model_selector.mode == "genetic"
    icselector.set_mode("invalid")
    assert icselector._model_selector.mode == "genetic"


def test_icselector_get_by_channel(icselector):
    mech = icselector.get("K_Tst")
    assert mech is not None


def test_icselector_get_by_gene(icselector):
    # Kcna1 -> Kv1.1 -> Kv1_1/K_Tst
    mech = icselector.get("Kcna1")
    assert mech is not None


def test_icselector_get_unknown_returns_none(icselector):
    assert icselector.get("does_not_exist") is None


def test_icselector_select(icselector):
    icselector.select("pas")
    mech = icselector.get("pas")
    assert mech.is_selected() is True


def test_icselector_get_mechanisms(icselector):
    icselector.select("pas")
    mechs = icselector.get_mechanisms(selected_only=True)
    assert "pas" in mechs


def test_icselector_get_cell_config_from_ttype(icselector):
    parameters, mechanisms, distributions, nexus_keys = icselector.get_cell_config_from_ttype(
        ["Test Ttype_1"]
    )
    assert isinstance(parameters, list)
    assert isinstance(mechanisms, list)
    assert isinstance(distributions, list)
    assert isinstance(nexus_keys, list)
    assert len(parameters) > 0


def test_icselector_get_gene_mapping(icselector):
    icselector.get_cell_config_from_ttype(["Test Ttype_1"])
    genes = icselector.get_gene_mapping()
    assert isinstance(genes, dict)


def test_icselector_get_selected_cell_types(icselector):
    icselector.get_cell_config_from_ttype(["Test Ttype_1"])
    cell_types = icselector.get_selected_cell_types()
    assert len(cell_types) > 0


def test_icselector_warns_on_unmapped_gene(icselector, caplog):
    # Kcnc1 maps to Kv3.1, which only maps to SKv3_1 in the test ic_map,
    # so no unmapped case exists there; force one by selecting a gene that
    # has no corresponding channel/mechanism entry at all.
    icselector._gene_selector.selected_genes = {
        "UnmappedGene": {"channel": "n/a", "distribution": None, "gbar_max": 0}
    }
    icselector._set_parameters_from_ttype()
    assert "Could not determine mechanism for gene" in caplog.text


def test_icselector_get_cell_config_from_ttype_unknown_distribution(ic_map, tmp_path, caplog):
    # Force the "pas" mechanism to be configured with a non-uniform distribution
    # name that has no corresponding entry in ic_map["distributions"], to hit
    # the "Unknown distribution" warning path.
    ic_map["mechanism_parameters"]["pas"]["distribution"] = "some_unknown_distribution"
    tmp_ic_map_path = tmp_path / "ic_map.json"
    tmp_ic_map_path.write_text(json.dumps(ic_map))

    selector = ICSelector(
        ic_map_path=tmp_ic_map_path,
        gene_map_path=ICSELECTOR_DATA / "gene_map.csv",
    )
    _, _, distributions, _ = selector.get_cell_config_from_ttype(["Test Ttype_1"])

    assert any(d["name"] == "some_unknown_distribution" for d in distributions)
    assert "Unknown distribution" in caplog.text


def test_model_selector_select_with_requires(ic_map):
    ic_map["mechanisms"]["SKv3_1"]["requires"] = ["K_Tst"]
    selector = ModelSelector(ic_map)
    mech = selector.select("SKv3_1")
    assert mech.is_selected() is True
    req_mech = selector.get("K_Tst")
    assert req_mech.is_selected() is True


def test_model_selector_check_mech_none():
    ic_map_local = {
        "mechanisms": {},
        "channels": {},
    }
    selector = ModelSelector(ic_map_local)
    assert selector._check_mech(None) is False
