"""Versioned optimization artifacts for BluePyEModel working-directory staging."""

from collections.abc import Mapping

from bluepyemodel.preprocessing.parameters import build_params_definition
from bluepyemodel.preprocessing.recipes import build_optimization_recipe
from bluepyemodel.preprocessing.recipes import update_pipeline_settings
from bluepyemodel.preprocessing.schemas import NormalizedIonChannelModel
from bluepyemodel.preprocessing.schemas import OptimizationArtifactInput
from bluepyemodel.preprocessing.schemas import OptimizationArtifacts
from bluepyemodel.preprocessing.schemas import ParamsDefinitionInput

TASK2_CONFIG_CONTRACT_VERSION = "task2-config-v2"
TASK2_ARTIFACT_CONTRACT_VERSION = "task2-artifacts-v1"


def build_optimization_artifacts(
    artifact_input: OptimizationArtifactInput,
    normalized_ion_channel_models: Mapping[str, NormalizedIonChannelModel],
) -> OptimizationArtifacts:
    """Compile validated local preprocessing inputs into the artifact bundle.

    This function is deliberately pure: entity resolution, downloads, filesystem staging,
    mechanism compilation, optimization, and registration remain outside the boundary.
    """
    if artifact_input.config_contract_version != TASK2_CONFIG_CONTRACT_VERSION:
        msg = (
            "Unsupported optimization configuration contract version: "
            f"{artifact_input.config_contract_version!r}; expected "
            f"{TASK2_CONFIG_CONTRACT_VERSION!r}."
        )
        raise ValueError(msg)

    params_input = ParamsDefinitionInput(
        parameters_selection=artifact_input.parameters_selection,
        distance_dependent_distributions=artifact_input.distance_dependent_distributions,
    )
    params = build_params_definition(
        params_input,
        normalized_ion_channel_models,
        morphology_capabilities=artifact_input.morphology_capabilities,
    )
    recipes = build_optimization_recipe(
        artifact_input.emodel,
        artifact_input.mtype,
        artifact_input.morphology_filename,
    )
    update_pipeline_settings(
        recipes,
        emodel=artifact_input.emodel,
        overrides=artifact_input.pipeline_settings_overrides,
    )
    return OptimizationArtifacts(
        config_contract_version=artifact_input.config_contract_version,
        artifact_contract_version=TASK2_ARTIFACT_CONTRACT_VERSION,
        params=params,
        recipes=recipes,
    )
