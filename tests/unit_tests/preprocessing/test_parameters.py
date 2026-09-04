"""Coverage for preprocessing parameter compilation edge cases."""

from types import SimpleNamespace

import pytest

from bluepyemodel.preprocessing import ParamsDefinitionInput
from bluepyemodel.preprocessing import build_params_definition
from bluepyemodel.preprocessing import normalize_ion_channel_model
from bluepyemodel.preprocessing.schemas import DEFAULT_SECTION_LIST_CATALOG
from bluepyemodel.preprocessing.schemas import CustomDistanceDependentDistribution
from bluepyemodel.preprocessing.schemas import GlobalParameterSelection
from bluepyemodel.preprocessing.schemas import IonChannelModelRef
from bluepyemodel.preprocessing.schemas import MechanismRegionSelection
from bluepyemodel.preprocessing.schemas import MorphologyCapabilities
from bluepyemodel.preprocessing.schemas import OptimizationValue
from bluepyemodel.preprocessing.schemas import OptimizationValueMode
from bluepyemodel.preprocessing.schemas import ParameterSelection
from bluepyemodel.preprocessing.schemas import ParametersSelection
from bluepyemodel.preprocessing.schemas import PhysicalSectionListName
from bluepyemodel.preprocessing.schemas import SectionListName


def _entity(
    *,
    suffix="NaTg",
    entity_id="icm-1",
    useion=None,
    range_entries=None,
    global_entries=None,
    neuron_block=True,
):
    block = None
    if neuron_block:
        block = SimpleNamespace(
            range=(
                range_entries
                if range_entries is not None
                else [{"gNa": "S/cm2"}, {"variable": "vshift", "units": "mV"}]
            ),
            global_=(
                global_entries
                if global_entries is not None
                else [{"variable": "ena", "units": "mV"}]
            ),
            useion=useion,
        )
    return SimpleNamespace(
        id=entity_id,
        name="Sodium channel",
        nmodl_suffix=suffix,
        is_stochastic=False,
        is_ljp_corrected=False,
        temperature_celsius=34,
        neuron_block=block,
    )


def _selection(**overrides):
    reference = IonChannelModelRef(id_str="icm-1")
    data = {
        "ion_channel_models": (reference,),
        "mechanism_regions": {
            SectionListName.apical: (
                MechanismRegionSelection(
                    ion_channel_model=reference,
                    parameters={
                        "gNa": ParameterSelection(
                            value=OptimizationValue(
                                mode=OptimizationValueMode.bounds,
                                bounds=(0.0, 1.0),
                            ),
                            distribution="decay",
                        ),
                    },
                ),
            ),
        },
        "global_parameters": {
            "v_init": GlobalParameterSelection(value=OptimizationValue(value=-80.0)),
        },
        "base_parameters": {
            SectionListName.all: {
                "Ra": ParameterSelection(value=OptimizationValue(value=100.0)),
                "g_pas": ParameterSelection(
                    value=OptimizationValue(mode=OptimizationValueMode.bounds, bounds=(1e-5, 6e-5))
                ),
            }
        },
        "distribution_parameters": {
            "decay": {"lambda": OptimizationValue(value=0.1)},
        },
    }
    data.update(overrides)
    return ParametersSelection(**data)


def _params_input(selection=None, distributions=None):
    return ParamsDefinitionInput(
        parameters_selection=selection or _selection(),
        distance_dependent_distributions=distributions
        or {
            "decay": CustomDistanceDependentDistribution(
                name="decay",
                function="math.exp(-{distance}/{lambda})*{value}",
                parameters=["lambda"],
            )
        },
    )


def test_normalize_ion_channel_model_accepts_alternate_shapes_and_deduplicates():
    entity = _entity(
        range_entries=[
            SimpleNamespace(name="gNa", units="S/cm2"),
            {"gNa": "S/cm2"},
            {"variable": "", "units": "mV"},
        ],
        global_entries=None,
    )
    delattr(entity.neuron_block, "global_")
    setattr(entity.neuron_block, "global", [{"ena": "mV"}])
    model = normalize_ion_channel_model(entity)
    assert {variable.source_name for variable in model.range_variables} == {"gNa"}
    assert {variable.source_name for variable in model.global_variables} == {"ena"}
    assert model.find_variable("missing") is None
    assert model.find_variable("gNa").name == "gNa_NaTg"


def test_normalize_ion_channel_model_rejects_incomplete_metadata():
    with pytest.raises(ValueError, match="nmodl_suffix"):
        normalize_ion_channel_model(SimpleNamespace(id="x", neuron_block=object()))
    with pytest.raises(ValueError, match="no entity ID"):
        normalize_ion_channel_model(SimpleNamespace(nmodl_suffix="NaTg", neuron_block=object()))
    with pytest.raises(ValueError, match="neuron_block"):
        normalize_ion_channel_model(_entity(neuron_block=False))


def test_build_params_definition_rejects_invalid_bounds_and_fixed_values():
    with pytest.raises(ValueError, match="must be finite"):
        build_params_definition(
            _params_input(
                selection=_selection(mechanism_regions={}, distribution_parameters={}),
                distributions={},
            ),
            {},
            bounds_fallbacks={"broken": (float("nan"), 1.0)},
        )

    with pytest.raises(ValueError, match="Lower bound exceeds"):
        build_params_definition(
            _params_input(
                selection=_selection(mechanism_regions={}, distribution_parameters={}),
                distributions={},
            ),
            {},
            bounds_fallbacks={"broken": (2.0, 1.0)},
        )

    no_bounds = ParametersSelection.model_construct(
        ion_channel_models=(),
        mechanism_regions={},
        global_parameters={
            "v_init": GlobalParameterSelection(
                value=OptimizationValue(mode=OptimizationValueMode.bounds)
            )
        },
        base_parameters={},
        distribution_parameters={},
    )
    with pytest.raises(ValueError, match="no bounds"):
        build_params_definition(ParamsDefinitionInput(parameters_selection=no_bounds), {})

    fixed_missing = ParametersSelection.model_construct(
        ion_channel_models=(),
        mechanism_regions={},
        global_parameters={
            "v_init": GlobalParameterSelection.model_construct(
                value=OptimizationValue.model_construct(
                    mode=OptimizationValueMode.fixed,
                    value=None,
                    bounds=None,
                ),
                ion_channel_model=None,
            )
        },
        base_parameters={},
        distribution_parameters={},
    )
    with pytest.raises(ValueError, match="has no value"):
        build_params_definition(ParamsDefinitionInput(parameters_selection=fixed_missing), {})


def test_build_params_definition_uses_fallback_bounds_and_warns_on_overlap(caplog):
    selection = _selection(
        base_parameters={
            SectionListName.all: {
                "cm": ParameterSelection(
                    value=OptimizationValue(mode=OptimizationValueMode.bounds),
                )
            },
            SectionListName.apical: {
                "cm": ParameterSelection(value=OptimizationValue(value=2.0)),
            },
        },
        mechanism_regions={},
        distribution_parameters={},
    )
    with caplog.at_level("WARNING"):
        params = build_params_definition(
            _params_input(selection, {}),
            {},
            bounds_fallbacks={"cm": (0.5, 1.5)},
        )
    assert {"name": "cm", "val": [0.5, 1.5]} in params["parameters"]["all"]
    assert any("Overlapping parameter rows" in message for message in caplog.messages)


def test_build_params_definition_rejects_missing_models_and_duplicate_assignments():
    selection = _selection()
    with pytest.raises(ValueError, match="No normalized metadata"):
        build_params_definition(_params_input(selection), {})

    reference = IonChannelModelRef(id_str="icm-1")
    duplicate = _selection(
        mechanism_regions={
            SectionListName.apical: (
                MechanismRegionSelection(ion_channel_model=reference),
                MechanismRegionSelection(ion_channel_model=reference),
            )
        },
        distribution_parameters={},
    )
    normalized = {"icm-1": normalize_ion_channel_model(_entity())}
    with pytest.raises(ValueError, match="more than once"):
        build_params_definition(_params_input(duplicate, {}), normalized)


def test_build_params_definition_validates_global_and_mechanism_parameters():
    reference = IonChannelModelRef(id_str="icm-1")
    normalized = {"icm-1": normalize_ion_channel_model(_entity())}

    missing_global_source = _selection(
        global_parameters={
            "ena": GlobalParameterSelection(
                value=OptimizationValue(value=50.0),
                ion_channel_model=IonChannelModelRef(id_str="missing"),
            )
        },
        ion_channel_models=(reference, IonChannelModelRef(id_str="missing")),
        mechanism_regions={},
        distribution_parameters={},
    )
    with pytest.raises(ValueError, match="No normalized metadata for global"):
        build_params_definition(_params_input(missing_global_source, {}), normalized)

    bad_global = _selection(
        global_parameters={
            "gNa": GlobalParameterSelection(
                value=OptimizationValue(value=1.0),
                ion_channel_model=reference,
            )
        },
        mechanism_regions={},
        distribution_parameters={},
    )
    with pytest.raises(ValueError, match="GLOBAL variable"):
        build_params_definition(_params_input(bad_global, {}), normalized)

    unknown_param = _selection(
        mechanism_regions={
            SectionListName.apical: (
                MechanismRegionSelection(
                    ion_channel_model=reference,
                    parameters={"unknown": ParameterSelection(value=OptimizationValue(value=1.0))},
                ),
            )
        },
        distribution_parameters={},
    )
    with pytest.raises(ValueError, match="is not declared"):
        build_params_definition(_params_input(unknown_param, {}), normalized)

    global_as_range = _selection(
        mechanism_regions={
            SectionListName.apical: (
                MechanismRegionSelection(
                    ion_channel_model=reference,
                    parameters={"ena": ParameterSelection(value=OptimizationValue(value=50.0))},
                ),
            )
        },
        distribution_parameters={},
    )
    with pytest.raises(ValueError, match="must be configured in global_parameters"):
        build_params_definition(_params_input(global_as_range, {}), normalized)


def test_build_params_definition_rejects_distribution_and_morphology_errors():
    reference = IonChannelModelRef(id_str="icm-1")
    normalized = {"icm-1": normalize_ion_channel_model(_entity())}

    undeclared = _selection(
        distribution_parameters={"missing": {"lambda": OptimizationValue(value=1.0)}},
        mechanism_regions={},
    )
    with pytest.raises(ValueError, match="is not declared"):
        build_params_definition(_params_input(undeclared, {}), {})

    unknown_dist_param = _selection(
        distribution_parameters={"decay": {"unknown": OptimizationValue(value=1.0)}},
        mechanism_regions={},
    )
    with pytest.raises(ValueError, match="undeclared parameters"):
        build_params_definition(_params_input(unknown_dist_param), {})

    missing_values = _selection(
        distribution_parameters={},
        mechanism_regions={
            SectionListName.apical: (
                MechanismRegionSelection(
                    ion_channel_model=reference,
                    parameters={
                        "gNa": ParameterSelection(
                            value=OptimizationValue(value=1.0),
                            distribution="decay",
                        )
                    },
                ),
            )
        },
    )
    with pytest.raises(ValueError, match="missing values"):
        build_params_definition(_params_input(missing_values), normalized)

    myelinated = _selection(
        mechanism_regions={},
        distribution_parameters={},
        base_parameters={
            SectionListName.myelinated: {
                "cm": ParameterSelection(value=OptimizationValue(value=0.02))
            }
        },
    )
    with pytest.raises(ValueError, match="no myelinated section list"):
        build_params_definition(
            _params_input(myelinated, {}),
            {},
            morphology_capabilities=MorphologyCapabilities(has_myelinated=False),
        )
    with pytest.raises(ValueError, match="did not establish a myelinated section list"):
        build_params_definition(
            _params_input(myelinated, {}),
            {},
            morphology_capabilities=MorphologyCapabilities(has_myelinated=None),
        )

    with pytest.raises(TypeError, match="MorphologyCapabilities"):
        build_params_definition(
            _params_input(_selection(mechanism_regions={}, distribution_parameters={}), {}),
            {},
            morphology_capabilities=object(),
        )

    unavailable = _selection(
        mechanism_regions={},
        distribution_parameters={},
        base_parameters={
            SectionListName.apical: {"cm": ParameterSelection(value=OptimizationValue(value=1.0))}
        },
    )
    with pytest.raises(ValueError, match="no source sections"):
        build_params_definition(
            _params_input(unavailable, {}),
            {},
            morphology_capabilities=MorphologyCapabilities(
                available_physical_sections=(PhysicalSectionListName.somatic,),
            ),
        )


def test_build_params_definition_emits_global_mechanism_variable_and_skips_duplicate_pas():
    reference = IonChannelModelRef(id_str="icm-1")
    normalized = {"icm-1": normalize_ion_channel_model(_entity())}
    selection = _selection(
        global_parameters={
            "ena": GlobalParameterSelection(
                value=OptimizationValue(value=50.0),
                ion_channel_model=reference,
            )
        },
        mechanism_regions={
            SectionListName.all: (MechanismRegionSelection(ion_channel_model=reference),)
        },
        base_parameters={
            SectionListName.all: {
                "g_pas": ParameterSelection(value=OptimizationValue(value=1e-5)),
            }
        },
        distribution_parameters={},
    )
    params = build_params_definition(_params_input(selection, {}), normalized)
    assert {"name": "ena_NaTg", "val": 50.0} in params["parameters"]["global"]
    assert params["mechanisms"]["all"]["mech"] == ["NaTg", "pas"]


def test_build_params_definition_accepts_myelinated_when_capability_is_true():
    selection = _selection(
        mechanism_regions={},
        distribution_parameters={},
        base_parameters={
            SectionListName.myelinated: {
                "cm": ParameterSelection(value=OptimizationValue(value=0.02))
            }
        },
    )
    params = build_params_definition(
        _params_input(selection, {}),
        {},
        morphology_capabilities=MorphologyCapabilities(has_myelinated=True),
    )
    assert params["parameters"]["myelinated"] == [{"name": "cm", "val": 0.02}]


def test_build_params_definition_rejects_undeclared_distribution_on_base_parameter():
    selection = _selection(
        mechanism_regions={},
        distribution_parameters={},
        base_parameters={
            SectionListName.all: {
                "Ra": ParameterSelection(
                    value=OptimizationValue(value=100.0),
                    distribution="missing",
                )
            }
        },
    )
    with pytest.raises(ValueError, match="undeclared distribution"):
        build_params_definition(_params_input(selection, {}), {})


def test_build_params_definition_rejects_unsupported_locations_via_construct():
    from bluepyemodel.preprocessing import parameters as parameters_module

    with pytest.raises(ValueError, match="Unsupported regional parameter location"):
        parameters_module._validate_location("unsupported")

    assert parameters_module._location_sort_key("unsupported") == (0, "unsupported")  # type: ignore[arg-type]


def test_build_params_definition_skips_non_overlapping_rows_and_duplicate_pas(caplog):
    pas_ref = IonChannelModelRef(id_str="pas-1")
    normalized = {
        "pas-1": normalize_ion_channel_model(
            _entity(suffix="pas", entity_id="pas-1", range_entries=[], global_entries=[])
        ),
    }
    selection = _selection(
        ion_channel_models=(pas_ref,),
        mechanism_regions={
            SectionListName.all: (MechanismRegionSelection(ion_channel_model=pas_ref),),
        },
        base_parameters={
            SectionListName.apical: {"cm": ParameterSelection(value=OptimizationValue(value=1.0))},
            SectionListName.basal: {"cm": ParameterSelection(value=OptimizationValue(value=2.0))},
            SectionListName.all: {
                "g_pas": ParameterSelection(value=OptimizationValue(value=1e-5)),
            },
        },
        distribution_parameters={},
    )
    with caplog.at_level("WARNING"):
        params = build_params_definition(_params_input(selection, {}), normalized)
    assert params["parameters"]["apical"] == [{"name": "cm", "val": 1.0}]
    assert params["parameters"]["basal"] == [{"name": "cm", "val": 2.0}]
    assert params["mechanisms"]["all"]["mech"] == ["pas"]
    assert not any("Overlapping parameter rows" in message for message in caplog.messages)


def test_to_legacy_distributions_preserves_non_default_soma_ref_location():
    from bluepyemodel.preprocessing import parameters as parameters_module

    legacy = parameters_module._to_legacy_distributions(
        [
            {
                "name": "custom",
                "function": "{value}",
                "parameters": ["lambda"],
                "soma_ref_location": 0.25,
            },
            {
                "name": "uniform",
                "function": None,
                "soma_ref_location": 0.5,
            },
        ]
    )
    assert legacy["custom"]["soma_ref_location"] == 0.25
    assert "soma_ref_location" not in legacy["uniform"]
    assert legacy["custom"]["parameters"] == ["lambda"]
