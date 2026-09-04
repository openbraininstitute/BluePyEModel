"""Tests for optimization artifact compilation."""

import json
from types import SimpleNamespace

import pytest

from bluepyemodel.preprocessing import (
    PARAMS_ARTIFACT_PATH,
    RECIPES_ARTIFACT_PATH,
    TASK2_ARTIFACT_CONTRACT_VERSION,
    TASK2_CONFIG_CONTRACT_VERSION,
    OptimizationArtifactInput,
    ParamsDefinitionInput,
    build_optimization_artifacts,
    build_optimization_recipe,
    build_params_definition,
    normalize_ion_channel_model,
)
from bluepyemodel.preprocessing.schemas import (
    DEFAULT_SECTION_LIST_CATALOG,
    CustomDistanceDependentDistribution,
    GlobalParameterSelection,
    IonChannelModelRef,
    MechanismRegionSelection,
    OptimizationValue,
    ParameterSelection,
    ParametersSelection,
)
from bluepyemodel.preprocessing.recipes import update_pipeline_settings


def _model_entity(useion=None):
    return SimpleNamespace(
        id="icm-1",
        name="Sodium channel",
        nmodl_suffix="NaTg",
        is_stochastic=False,
        is_ljp_corrected=False,
        temperature_celsius=34,
        neuron_block=SimpleNamespace(
            range=[{"gNa": "S/cm2"}, {"variable": "vshift", "units": "mV"}],
            global_=[{"variable": "ena", "units": "mV"}],
            useion=useion,
        ),
    )


def _compiler_fixture():
    reference = IonChannelModelRef(id_str="icm-1")
    selection = ParametersSelection(
        ion_channel_models=[reference],
        mechanism_regions={
            "apical": [
                MechanismRegionSelection(
                    ion_channel_model=reference,
                    parameters={
                        "gNa": ParameterSelection(
                            value=OptimizationValue(mode="bounds", bounds=[0.0, 1.0]),
                            distribution="decay",
                        ),
                    },
                ),
            ],
        },
        global_parameters={
            "v_init": GlobalParameterSelection(value=OptimizationValue(value=-80.0)),
        },
        base_parameters={
            "all": {
                "Ra": ParameterSelection(value=OptimizationValue(value=100.0)),
                "g_pas": ParameterSelection(value=OptimizationValue(mode="bounds")),
                "e_pas": ParameterSelection(value=OptimizationValue(mode="bounds")),
            }
        },
        distribution_parameters={
            "decay": {
                "lambda": OptimizationValue(value=0.1),
            },
        },
    )
    custom_distributions = {
        "decay": CustomDistanceDependentDistribution(
            name="decay",
            function="math.exp(-{distance}/{lambda})*{value}",
            parameters=["lambda"],
        ),
    }
    params_input = ParamsDefinitionInput(
        parameters_selection=selection,
        distance_dependent_distributions=custom_distributions,
    )
    artifact_input = OptimizationArtifactInput(
        config_contract_version=TASK2_CONFIG_CONTRACT_VERSION,
        emodel="test",
        parameters_selection=selection,
        distance_dependent_distributions=custom_distributions,
        pipeline_settings_overrides={"optimiser": "MO-CMA", "max_ngen": 10},
        mtype="L5PC",
        morphology_filename="morphology-1.swc",
    )
    normalized = {"icm-1": normalize_ion_channel_model(_model_entity())}
    return params_input, artifact_input, normalized


def test_build_optimization_recipe_uses_contract_paths():
    recipe = build_optimization_recipe("L5PC", "L5PC", "morph.swc")
    assert recipe["L5PC"]["params"] == "config/params/params.json"
    assert recipe["L5PC"]["features"] == "config/features/L5PC.json"


def test_build_params_definition_compiles_mechanisms_and_distributions():
    params_input, _, normalized = _compiler_fixture()
    params = build_params_definition(params_input, normalized)
    assert params["mechanisms"]["apical"] == {"mech": ["NaTg"]}
    assert params["mechanisms"]["all"] == {"mech": ["pas"]}
    assert params["parameters"]["apical"] == [
        {"name": "gNa_NaTg", "val": [0.0, 1.0], "dist": "decay"}
    ]
    assert {"name": "Ra", "val": 100.0} in params["parameters"]["all"]
    assert params["parameters"]["global"] == [{"name": "v_init", "val": -80.0}]
    assert params["parameters"]["distribution_decay"] == [{"name": "lambda", "val": 0.1}]


def test_build_params_definition_emits_legacy_distribution_mapping():
    params_input, _, normalized = _compiler_fixture()
    params = build_params_definition(params_input, normalized)
    assert params["distributions"] == {
        "decay": {
            "fun": "math.exp(-{distance}/{lambda})*{value}",
            "parameters": ["lambda"],
        }
    }
    assert "uniform" not in params["distributions"]
    assert "morphology" not in params


def test_reversal_potentials_are_dropped_without_a_matching_ion():
    params_input, _, normalized = _compiler_fixture()
    params_input.parameters_selection.base_parameters["somatic"] = {
        "ek": ParameterSelection(value=OptimizationValue(value=-90.0)),
    }
    params = build_params_definition(params_input, normalized)
    assert "somatic" not in params["parameters"]


def test_reversal_potentials_are_kept_when_the_ion_is_assigned():
    params_input, _, normalized = _compiler_fixture()
    reference = IonChannelModelRef(id_str="icm-1")
    params_input.parameters_selection.mechanism_regions["somatic"] = [
        MechanismRegionSelection(ion_channel_model=reference),
    ]
    params_input.parameters_selection.base_parameters["somatic"] = {
        "ek": ParameterSelection(value=OptimizationValue(value=-90.0)),
    }
    normalized["icm-1"] = normalize_ion_channel_model(_model_entity(useion=[{"ion_name": "K"}]))
    params = build_params_definition(params_input, normalized)
    assert params["parameters"]["somatic"] == [{"name": "ek", "val": -90.0}]


def test_normalize_ion_channel_model_reads_useion_names():
    model = normalize_ion_channel_model(
        _model_entity(useion=[{"ion_name": " Na "}, {"ion_name": "K"}])
    )
    assert model.ion_names == frozenset({"na", "k"})


def test_recipe_multiloc_map_excludes_the_all_alias():
    recipe = build_optimization_recipe("L5PC", "L5PC", "morph.swc")
    multiloc_map = recipe["L5PC"]["multiloc_map"]
    assert "all" not in multiloc_map
    assert "alldend" in multiloc_map
    assert "all" in DEFAULT_SECTION_LIST_CATALOG.to_alias_expansions()


def test_artifact_bundle_has_versioned_relative_paths_and_writes_json(tmp_path):
    _, artifact_input, normalized = _compiler_fixture()
    artifacts = build_optimization_artifacts(artifact_input, normalized)
    artifacts.write(tmp_path)

    assert artifacts.artifact_contract_version == TASK2_ARTIFACT_CONTRACT_VERSION
    assert artifacts.params_path == PARAMS_ARTIFACT_PATH
    assert artifacts.recipes_path == RECIPES_ARTIFACT_PATH
    assert (tmp_path / PARAMS_ARTIFACT_PATH).exists()
    assert (tmp_path / RECIPES_ARTIFACT_PATH).exists()
    assert (
        json.loads((tmp_path / RECIPES_ARTIFACT_PATH).read_text())["test"]["params"]
        == PARAMS_ARTIFACT_PATH
    )


def test_artifact_builder_rejects_unsupported_config_contract():
    _, artifact_input, normalized = _compiler_fixture()
    invalid_input = artifact_input.model_copy(update={"config_contract_version": "task2-config-v0"})
    with pytest.raises(ValueError, match="contract version"):
        build_optimization_artifacts(invalid_input, normalized)


def test_update_pipeline_settings_skips_none_values():
    recipes = {"emodel": {"pipeline_settings": {"optimiser": "MO-CMA"}}}
    update_pipeline_settings(recipes, emodel="emodel", overrides={"max_ngen": 50, "seed": None})
    assert recipes["emodel"]["pipeline_settings"] == {"optimiser": "MO-CMA", "max_ngen": 50}
