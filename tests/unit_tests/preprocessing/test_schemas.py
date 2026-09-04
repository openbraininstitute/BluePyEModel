"""Coverage for preprocessing schema validators and catalogue helpers."""

import math

import pytest

from bluepyemodel.preprocessing.schemas import DEFAULT_SECTION_LIST_CATALOG
from bluepyemodel.preprocessing.schemas import AxonModifier
from bluepyemodel.preprocessing.schemas import CustomDistanceDependentDistribution
from bluepyemodel.preprocessing.schemas import DistanceDependentDistribution
from bluepyemodel.preprocessing.schemas import GlobalParameterSelection
from bluepyemodel.preprocessing.schemas import IonChannelModelRef
from bluepyemodel.preprocessing.schemas import MechanismRegionSelection
from bluepyemodel.preprocessing.schemas import OptimizationValue
from bluepyemodel.preprocessing.schemas import OptimizationValueMode
from bluepyemodel.preprocessing.schemas import ParameterSelection
from bluepyemodel.preprocessing.schemas import ParametersSelection
from bluepyemodel.preprocessing.schemas import PhysicalSectionListName
from bluepyemodel.preprocessing.schemas import SectionListAvailability
from bluepyemodel.preprocessing.schemas import SectionListCatalog
from bluepyemodel.preprocessing.schemas import SectionListChoice
from bluepyemodel.preprocessing.schemas import SectionListDefinition
from bluepyemodel.preprocessing.schemas import SectionListName


def test_section_list_definition_validation_errors():
    with pytest.raises(ValueError, match="at least one section list"):
        SectionListDefinition(
            name=SectionListName.somatic,
            label="Soma",
            description="soma",
            expanded_sections=(),
        )
    with pytest.raises(ValueError, match="duplicate expanded"):
        SectionListDefinition(
            name=SectionListName.alldend,
            label="Dendrites",
            description="d",
            expanded_sections=(
                PhysicalSectionListName.apical,
                PhysicalSectionListName.apical,
            ),
            is_composite=True,
        )
    with pytest.raises(ValueError, match="must expand to multiple"):
        SectionListDefinition(
            name=SectionListName.alldend,
            label="Dendrites",
            description="d",
            expanded_sections=(PhysicalSectionListName.apical,),
            is_composite=True,
        )
    with pytest.raises(ValueError, match="may only expand to itself"):
        SectionListDefinition(
            name=SectionListName.myelinated,
            label="Myelin",
            description="m",
            expanded_sections=(PhysicalSectionListName.axonal,),
            requires_myelinated=True,
        )
    with pytest.raises(ValueError, match="requiring myelinated"):
        SectionListDefinition(
            name=SectionListName.axonal,
            label="Axon",
            description="a",
            expanded_sections=(PhysicalSectionListName.axonal,),
            requires_myelinated=True,
        )
    with pytest.raises(ValueError, match="must not contain myelinated"):
        SectionListDefinition(
            name=SectionListName.all,
            label="All",
            description="a",
            expanded_sections=(
                PhysicalSectionListName.somatic,
                PhysicalSectionListName.myelinated,
            ),
            is_composite=True,
        )


def test_section_list_choice_validation_and_enabled_alias():
    with pytest.raises(ValueError, match="cannot have unavailable status"):
        SectionListChoice(
            name=SectionListName.somatic,
            label="Soma",
            description="s",
            available=True,
            availability=SectionListAvailability.unavailable,
        )
    with pytest.raises(ValueError, match="must have unavailable status"):
        SectionListChoice(
            name=SectionListName.somatic,
            label="Soma",
            description="s",
            available=False,
            availability=SectionListAvailability.available,
            disabled_reason="reason",
        )
    with pytest.raises(ValueError, match="cannot have a disabled reason"):
        SectionListChoice(
            name=SectionListName.somatic,
            label="Soma",
            description="s",
            available=True,
            disabled_reason="reason",
        )
    with pytest.raises(ValueError, match="needs a disabled reason"):
        SectionListChoice(
            name=SectionListName.somatic,
            label="Soma",
            description="s",
            available=False,
            availability=SectionListAvailability.unavailable,
        )
    choice = SectionListChoice(
        name=SectionListName.somatic,
        label="Soma",
        description="s",
    )
    assert choice.enabled is True


def test_section_list_catalog_validation_and_lookup_helpers():
    with pytest.raises(ValueError, match="unique names"):
        SectionListCatalog(
            definitions=(
                *DEFAULT_SECTION_LIST_CATALOG.definitions,
                DEFAULT_SECTION_LIST_CATALOG.definitions[0],
            )
        )
    with pytest.raises(ValueError, match="missing definitions"):
        SectionListCatalog(definitions=DEFAULT_SECTION_LIST_CATALOG.definitions[:3])

    catalog = DEFAULT_SECTION_LIST_CATALOG
    with pytest.raises(ValueError, match="Unsupported section-list name"):
        catalog.definition("not-a-section")  # type: ignore[arg-type]

    assert catalog.available(SectionListName.somatic) is True
    assert catalog.available(SectionListName.myelinated, axon_modifier=AxonModifier.none) is False
    assert (
        catalog.available(
            SectionListName.myelinated,
            axon_modifier=AxonModifier.replace_axon_legacy,
        )
        is False
    )
    assert (
        catalog.available(
            SectionListName.myelinated,
            axon_modifier=AxonModifier.replace_axon_with_taper,
        )
        is True
    )
    assert catalog.choice(SectionListName.somatic).enabled is True
    assert catalog.schema_choices()
    assert set(catalog.schema_availability_by_modifier()) == {
        modifier.value for modifier in AxonModifier
    }
    assert "all" in catalog.to_alias_expansions()
    assert "all" not in catalog.to_recipe_multiloc_map()


def test_distance_dependent_function_placeholder_validation():
    with pytest.raises(ValueError, match=r"\{value\}"):
        DistanceDependentDistribution(function="{distance}")
    with pytest.raises(ValueError, match=r"\{distance\}"):
        DistanceDependentDistribution(function="{value}")
    with pytest.raises(ValueError, match=r"\{constant\}"):
        CustomDistanceDependentDistribution(
            name="custom",
            function="{value}+{distance}",
            parameters=["constant"],
        )
    distribution = CustomDistanceDependentDistribution(
        name="custom",
        function="({value}+{distance}+{constant})",
        parameters=["constant"],
        soma_ref_location=0.1,
    )
    assert distribution.to_emc_dict()["parameters"] == ["constant"]


def test_optimization_value_validation():
    with pytest.raises(ValueError, match="must be finite"):
        OptimizationValue(value=math.inf)
    with pytest.raises(ValueError, match="bounds must be finite"):
        OptimizationValue(mode=OptimizationValueMode.bounds, bounds=(math.nan, 1.0))
    with pytest.raises(ValueError, match="must not exceed"):
        OptimizationValue(mode=OptimizationValueMode.bounds, bounds=(2.0, 1.0))
    with pytest.raises(ValueError, match="required when mode is 'fixed'"):
        OptimizationValue(mode=OptimizationValueMode.fixed, value=None)
    with pytest.raises(ValueError, match="Bounds cannot be provided"):
        OptimizationValue(mode=OptimizationValueMode.fixed, value=1.0, bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="cannot be provided when mode is 'bounds'"):
        OptimizationValue(mode=OptimizationValueMode.bounds, value=1.0, bounds=(0.0, 1.0))


def test_parameters_selection_reference_validation_and_defaults():
    with pytest.raises(ValueError, match="duplicate entity IDs"):
        ParametersSelection(
            ion_channel_models=(
                IonChannelModelRef(id_str="icm-1"),
                IonChannelModelRef(id_str="icm-1"),
            )
        )
    with pytest.raises(ValueError, match="must also be listed"):
        ParametersSelection(
            ion_channel_models=(IonChannelModelRef(id_str="icm-1"),),
            mechanism_regions={
                SectionListName.apical: (
                    MechanismRegionSelection(ion_channel_model=IonChannelModelRef(id_str="other")),
                )
            },
        )
    with pytest.raises(ValueError, match="Global parameter"):
        ParametersSelection(
            ion_channel_models=(IonChannelModelRef(id_str="icm-1"),),
            global_parameters={
                "ena": GlobalParameterSelection(
                    value=OptimizationValue(value=50.0),
                    ion_channel_model=IonChannelModelRef(id_str="other"),
                )
            },
        )
    mechanism_guard = ParametersSelection.model_construct(
        ion_channel_models=(IonChannelModelRef(id_str="icm-1"),),
        mechanism_regions={},
        global_parameters={},
        base_parameters={},
        distribution_parameters={},
    )
    object.__setattr__(mechanism_guard, "mechanism_regions", {"unsupported": ()})
    with pytest.raises(ValueError, match="Unsupported mechanism region"):
        mechanism_guard.validate_selection_references()

    base_guard = ParametersSelection.model_construct(
        ion_channel_models=(),
        mechanism_regions={},
        global_parameters={},
        base_parameters={},
        distribution_parameters={},
    )
    object.__setattr__(base_guard, "base_parameters", {"unsupported": {}})
    with pytest.raises(ValueError, match="Unsupported base parameter region"):
        base_guard.validate_selection_references()

    selection = ParametersSelection()
    assert "v_init" in selection.global_parameters
    assert SectionListName.all in selection.base_parameters
    assert ParameterSelection(value=OptimizationValue(value=1.0))
