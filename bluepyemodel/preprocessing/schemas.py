"""Preprocessing schemas, dataclasses, and related constants."""

import math
from dataclasses import dataclass
from enum import StrEnum
from enum import auto
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Self
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import model_validator

from bluepyemodel.utils.io import write_json


class AxonModifier(StrEnum):
    """Allowlisted BluePyEModel morphology modifier names."""

    replace_axon_with_taper = auto()
    replace_axon_legacy = auto()
    replace_axon_olfactory_bulb = auto()
    bluepyopt_replace_axon = auto()
    none = auto()


class SectionListAvailability(StrEnum):
    """Availability state exposed to the form for a section-list choice."""

    available = auto()
    unavailable = auto()
    unknown = auto()


class PhysicalSectionListName(StrEnum):
    """Concrete NEURON section lists present on a morphology."""

    somatic = auto()
    basal = auto()
    apical = auto()
    axonal = auto()
    myelinated = auto()


class SectionListName(StrEnum):
    """Primitive and composite BluePyEModel section-list names."""

    all = auto()
    alldend = auto()
    somadend = auto()
    allnoaxon = auto()
    somaxon = auto()
    allact = auto()
    somatic = auto()
    basal = auto()
    apical = auto()
    axonal = auto()
    myelinated = auto()


RegionalSectionListName: TypeAlias = SectionListName

PHYSICAL_SECTION_LIST_NAMES: tuple[PhysicalSectionListName, ...] = tuple(PhysicalSectionListName)

REGIONAL_SECTION_LIST_NAMES: tuple[RegionalSectionListName, ...] = (
    SectionListName.all,
    SectionListName.alldend,
    SectionListName.somadend,
    SectionListName.allnoaxon,
    SectionListName.somaxon,
    SectionListName.allact,
    SectionListName.somatic,
    SectionListName.basal,
    SectionListName.apical,
    SectionListName.axonal,
    SectionListName.myelinated,
)

AXON_MODIFIER_DESCRIPTIONS: dict[AxonModifier, str] = {
    AxonModifier.replace_axon_with_taper: (
        "Replace the source axon with a tapered AIS and one synthesized myelin section."
    ),
    AxonModifier.replace_axon_legacy: (
        "Use the legacy two-section axon replacement without synthesized myelin."
    ),
    AxonModifier.replace_axon_olfactory_bulb: (
        "Use the olfactory-bulb hillock, node, and myelin replacement."
    ),
    AxonModifier.bluepyopt_replace_axon: (
        "Use BluePyOpt's built-in two-section axon replacement without synthesized myelin."
    ),
    AxonModifier.none: (
        "Keep the imported morphology without an axon replacement; source myelination is unknown."
    ),
}


_COMPOSITE_SECTION_COUNT = 2


class SectionListDefinition(BaseModel):
    """Validated definition of one primitive or composite section-list choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[SectionListName, Field(title="Section-list name")]
    label: Annotated[str, Field(title="Section-list label")]
    description: Annotated[str, Field(title="Section-list description")]
    expanded_sections: Annotated[
        tuple[PhysicalSectionListName, ...],
        Field(
            title="Expanded section lists",
            description="Concrete NEURON section lists used by the BluePyEModel alias.",
        ),
    ]
    is_composite: Annotated[
        bool,
        Field(
            default=False,
            title="Composite section list",
            description="Whether this name expands to more than one concrete section list.",
        ),
    ] = False
    requires_myelinated: Annotated[
        bool,
        Field(
            default=False,
            title="Requires myelinated sections",
            description="Whether the choice requires a synthesized or source myelinated list.",
        ),
    ] = False
    display_order: Annotated[
        int,
        Field(
            default=0,
            title="Display order",
            description=(
                "Position of this choice in the form's region-card list. Independent from "
                "compilation order, which is always broad-to-narrow."
            ),
        ),
    ] = 0

    @model_validator(mode="after")
    def validate_expansion(self) -> Self:
        """Ensure aliases have stable, non-empty, non-duplicated expansions."""
        if not self.expanded_sections:
            msg = f"Section-list '{self.name}' must expand to at least one section list."
            raise ValueError(msg)
        if len(set(self.expanded_sections)) != len(self.expanded_sections):
            msg = f"Section-list '{self.name}' contains duplicate expanded section lists."
            raise ValueError(msg)
        if self.is_composite and len(self.expanded_sections) < _COMPOSITE_SECTION_COUNT:
            msg = f"Composite section-list '{self.name}' must expand to multiple section lists."
            raise ValueError(msg)
        if self.name == SectionListName.myelinated and self.expanded_sections != (
            PhysicalSectionListName.myelinated,
        ):
            msg = "The myelinated section list may only expand to itself."
            raise ValueError(msg)
        if (
            self.requires_myelinated
            and PhysicalSectionListName.myelinated not in self.expanded_sections
        ):
            msg = f"Section-list '{self.name}' is marked as requiring myelinated sections."
            raise ValueError(msg)
        if (
            PhysicalSectionListName.myelinated in self.expanded_sections
            and self.name != SectionListName.myelinated
        ):
            msg = "Combined section-list aliases must not contain myelinated sections."
            raise ValueError(msg)
        return self


class SectionListChoice(BaseModel):
    """A section-list definition annotated for form availability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[SectionListName, Field(title="Section-list name")]
    label: Annotated[str, Field(title="Section-list label")]
    description: Annotated[str, Field(title="Section-list description")]
    available: Annotated[
        bool,
        Field(
            default=True,
            title="Available",
            description="Whether the form may use this section list for the current modifier.",
        ),
    ] = True
    availability: Annotated[
        SectionListAvailability,
        Field(
            default=SectionListAvailability.available,
            title="Availability",
            description="Whether availability is known, unavailable, or dependent on source data.",
        ),
    ] = SectionListAvailability.available
    disabled_reason: Annotated[
        str | None,
        Field(
            default=None,
            title="Disabled reason",
            description="Reason shown when a section list is not selectable.",
        ),
    ] = None
    display_order: Annotated[
        int,
        Field(
            default=0,
            title="Display order",
            description="Position of this choice in the form's region-card list.",
        ),
    ] = 0

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        """Keep the boolean and descriptive availability state consistent."""
        if self.available and self.availability == SectionListAvailability.unavailable:
            msg = f"Available section-list '{self.name}' cannot have unavailable status."
            raise ValueError(msg)
        if not self.available and self.availability != SectionListAvailability.unavailable:
            msg = f"Unavailable section-list '{self.name}' must have unavailable status."
            raise ValueError(msg)
        if self.available and self.disabled_reason is not None:
            msg = f"Available section-list '{self.name}' cannot have a disabled reason."
            raise ValueError(msg)
        if not self.available and not self.disabled_reason:
            msg = f"Unavailable section-list '{self.name}' needs a disabled reason."
            raise ValueError(msg)
        return self

    @property
    def enabled(self) -> bool:
        """Alias used by form clients that call selectable choices enabled."""
        return self.available


def _default_section_list_definitions() -> tuple[SectionListDefinition, ...]:
    """Construct the canonical definitions after the model class is available."""
    return _make_default_section_list_definitions()


class SectionListCatalog(BaseModel):
    """Immutable, validated catalogue of primitive and composite section lists."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    definitions: Annotated[
        tuple[SectionListDefinition, ...],
        Field(
            default_factory=_default_section_list_definitions,
            title="Section-list definitions",
            description="Canonical BluePyEModel section-list names and alias expansions.",
        ),
    ]

    @model_validator(mode="after")
    def validate_catalog(self) -> Self:
        """Require exactly one definition for every canonical section-list name."""
        names = [definition.name for definition in self.definitions]
        if len(set(names)) != len(names):
            msg = "Section-list catalog definitions must have unique names."
            raise ValueError(msg)
        missing = set(REGIONAL_SECTION_LIST_NAMES) - set(names)
        if missing:
            msg = f"Section-list catalog is missing definitions: {sorted(missing)}."
            raise ValueError(msg)
        return self

    def definition(self, name: SectionListName) -> SectionListDefinition:
        """Return the validated definition for a section-list name."""
        for definition in self.definitions:
            if definition.name == name:
                return definition
        msg = f"Unsupported section-list name: {name}."
        raise ValueError(msg)

    def expand(self, name: SectionListName) -> tuple[PhysicalSectionListName, ...]:
        """Expand a canonical name into concrete NEURON section-list names."""
        return self.definition(name).expanded_sections

    def choice(
        self,
        name: SectionListName,
        *,
        axon_modifier: AxonModifier | str = AxonModifier.replace_axon_with_taper,
    ) -> SectionListChoice:
        """Return one form choice with modifier-specific myelin availability."""
        definition = self.definition(name)
        if name != SectionListName.myelinated:
            return SectionListChoice(
                name=definition.name,
                label=definition.label,
                description=definition.description,
                display_order=definition.display_order,
            )

        modifier = AxonModifier(axon_modifier)
        if modifier in {
            AxonModifier.replace_axon_legacy,
            AxonModifier.bluepyopt_replace_axon,
        }:
            return SectionListChoice(
                name=definition.name,
                label=definition.label,
                description=definition.description,
                available=False,
                availability=SectionListAvailability.unavailable,
                disabled_reason=(
                    f"The '{modifier.value}' modifier does not create a myelinated section list."
                ),
                display_order=definition.display_order,
            )
        if modifier == AxonModifier.none:
            return SectionListChoice(
                name=definition.name,
                label=definition.label,
                description=(
                    f"{definition.description} The source morphology may or may not provide it; "
                    "this first-release form does not inspect the morphology asset."
                ),
                available=False,
                availability=SectionListAvailability.unavailable,
                disabled_reason=(
                    "No replacement leaves source myelination unknown, so the myelinated "
                    "section list is unavailable without morphology preflight."
                ),
                display_order=definition.display_order,
            )
        return SectionListChoice(
            name=definition.name,
            label=definition.label,
            description=definition.description,
            display_order=definition.display_order,
        )

    def choices(
        self,
        *,
        axon_modifier: AxonModifier | str = AxonModifier.replace_axon_with_taper,
    ) -> list[SectionListChoice]:
        """Return all canonical choices in stable catalogue order."""
        return [
            self.choice(definition.name, axon_modifier=axon_modifier)
            for definition in self.definitions
        ]

    def available(
        self,
        name: SectionListName,
        *,
        axon_modifier: AxonModifier | str = AxonModifier.replace_axon_with_taper,
    ) -> bool:
        """Return whether a section-list choice can be selected for a modifier."""
        return self.choice(name, axon_modifier=axon_modifier).available

    def schema_choices(self) -> list[dict[str, object]]:
        """Return Pydantic-validated choice metadata for the default modifier."""
        return [choice.model_dump(mode="json") for choice in self.choices()]

    def schema_availability_by_modifier(self) -> dict[str, list[dict[str, object]]]:
        """Return validated choices for every allowlisted axon modifier."""
        return {
            modifier.value: [
                choice.model_dump(mode="json") for choice in self.choices(axon_modifier=modifier)
            ]
            for modifier in AxonModifier
        }

    def to_alias_expansions(self) -> dict[str, list[str]]:
        """Build the complete alias expansion map for schema and form metadata."""
        return {
            definition.name: list(definition.expanded_sections)
            for definition in self.definitions
            if definition.is_composite
        }

    def to_recipe_multiloc_map(self) -> dict[str, list[str]]:
        """Build the recipe map, excluding the built-in ``all`` alias."""
        return {
            definition.name: list(definition.expanded_sections)
            for definition in self.definitions
            if definition.is_composite and definition.name != SectionListName.all
        }


def _make_default_section_list_definitions() -> list[SectionListDefinition]:
    return [
        SectionListDefinition(
            name=SectionListName.all,
            label="All sections",
            description="Apical, basal, somatic, and axonal sections.",
            expanded_sections=(
                PhysicalSectionListName.apical,
                PhysicalSectionListName.basal,
                PhysicalSectionListName.somatic,
                PhysicalSectionListName.axonal,
            ),
            is_composite=True,
            display_order=0,
        ),
        SectionListDefinition(
            name=SectionListName.myelinated,
            label="Myelinated",
            description="Myelinated sections created by a compatible axon modifier.",
            expanded_sections=(PhysicalSectionListName.myelinated,),
            requires_myelinated=True,
            display_order=1,
        ),
        SectionListDefinition(
            name=SectionListName.somadend,
            label="Soma and dendrites",
            description="Apical, basal, and somatic sections.",
            expanded_sections=(
                PhysicalSectionListName.apical,
                PhysicalSectionListName.basal,
                PhysicalSectionListName.somatic,
            ),
            is_composite=True,
            display_order=2,
        ),
        SectionListDefinition(
            name=SectionListName.somatic,
            label="Somatic",
            description="Somatic sections only.",
            expanded_sections=(PhysicalSectionListName.somatic,),
            display_order=3,
        ),
        SectionListDefinition(
            name=SectionListName.axonal,
            label="Axonal",
            description="Axonal sections only.",
            expanded_sections=(PhysicalSectionListName.axonal,),
            display_order=4,
        ),
        SectionListDefinition(
            name=SectionListName.apical,
            label="Apical",
            description="Apical dendritic sections only.",
            expanded_sections=(PhysicalSectionListName.apical,),
            display_order=5,
        ),
        SectionListDefinition(
            name=SectionListName.basal,
            label="Basal",
            description="Basal dendritic sections only.",
            expanded_sections=(PhysicalSectionListName.basal,),
            display_order=6,
        ),
        SectionListDefinition(
            name=SectionListName.alldend,
            label="All dendrites",
            description="Apical and basal dendritic sections.",
            expanded_sections=(
                PhysicalSectionListName.apical,
                PhysicalSectionListName.basal,
            ),
            is_composite=True,
            display_order=7,
        ),
        SectionListDefinition(
            name=SectionListName.allnoaxon,
            label="All sections without axon",
            description="Apical, basal, and somatic sections.",
            expanded_sections=(
                PhysicalSectionListName.apical,
                PhysicalSectionListName.basal,
                PhysicalSectionListName.somatic,
            ),
            is_composite=True,
            display_order=8,
        ),
        SectionListDefinition(
            name=SectionListName.somaxon,
            label="Soma and axon",
            description="Axonal and somatic sections.",
            expanded_sections=(
                PhysicalSectionListName.axonal,
                PhysicalSectionListName.somatic,
            ),
            is_composite=True,
            display_order=9,
        ),
        SectionListDefinition(
            name=SectionListName.allact,
            label="All active sections",
            description="Apical, basal, somatic, and axonal sections for active mechanisms.",
            expanded_sections=(
                PhysicalSectionListName.apical,
                PhysicalSectionListName.basal,
                PhysicalSectionListName.somatic,
                PhysicalSectionListName.axonal,
            ),
            is_composite=True,
            display_order=10,
        ),
    ]


DEFAULT_SECTION_LIST_CATALOG = SectionListCatalog()


REGIONAL_PARAMETER_LOCATIONS: frozenset[RegionalSectionListName] = frozenset(
    REGIONAL_SECTION_LIST_NAMES
)


class StandardDistanceDependentDistributionName(StrEnum):
    """Built-in distance-dependent distribution identifiers."""

    uniform = auto()
    exp = auto()
    step = auto()
    exp_na_dend = auto()
    linear_hd_apic = auto()
    sigmoid_kad_apic = auto()
    linear_e_pas_apic = auto()
    linear_hdpas = auto()
    sigmoid_kad = auto()
    sigmoid_kdbm_apic = auto()


class DistanceDependentDistribution(BaseModel):
    """A BluePyEModel distance-dependent parameter transformation."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str | None,
        Field(
            default=None,
            min_length=1,
            title="Distribution name",
            description="Optional name used by BluePyEModel parameter definitions.",
        ),
    ] = None
    function: Annotated[
        str | None,
        Field(
            default=None,
            title="Distance function",
            description=(
                "Expression using {value} and {distance}; custom expressions may also use "
                "placeholders defined by the corresponding parameter configuration."
            ),
        ),
    ] = None
    soma_ref_location: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5
    parameters: Annotated[
        tuple[str, ...] | None,
        Field(
            default=None,
            title="Distribution parameters",
            description=(
                "Names of additional parameters that parametrize the function "
                "(excluding {value} and {distance}). Used by BluePyEModel's "
                "ParameterScaler."
            ),
        ),
    ] = None

    @model_validator(mode="after")
    def validate_function(self) -> Self:
        """Require functions to expose implicit and declared inputs.

        This only checks placeholder presence; it is not an AST validator or a
        sandbox. BluePyEModel evaluates the function string with Python ``eval()``
        at runtime (see ``bluepyemodel.model.model.define_distributions()``), so
        this validator must never be described as a security boundary.
        """
        if self.parameters and self.function is None:
            msg = "Distance-dependent distributions with parameters must define a function."
            raise ValueError(msg)
        # pylint: disable=unsupported-membership-test
        if self.function is not None and "{value}" not in self.function:
            msg = "Distance-dependent functions must contain the {value} placeholder."
            raise ValueError(msg)
        if self.function is not None and "{distance}" not in self.function:
            msg = "Distance-dependent functions must contain the {distance} placeholder."
            raise ValueError(msg)
        if self.function is not None:
            for parameter in self.parameters or []:
                placeholder = f"{{{parameter}}}"
                if placeholder not in self.function:
                    msg = (
                        f"Distance-dependent functions must contain the {placeholder} placeholder."
                    )
                    raise ValueError(msg)
        # pylint: enable=unsupported-membership-test
        return self

    def to_emc_dict(self, name: str | None = None) -> dict[str, Any]:
        """Convert the block to the legacy EMC distribution representation."""
        emc_dict: dict[str, Any] = {
            "name": name or self.name,
            "function": self.function,
            "soma_ref_location": self.soma_ref_location,
        }
        if self.parameters:
            emc_dict["parameters"] = list(self.parameters)
        return emc_dict


class UniformDistanceDependentDistribution(DistanceDependentDistribution):
    """Default uniform distance distribution used by EMC files."""

    name: Annotated[str, Field(default="uniform", frozen=True)] = "uniform"
    function: Annotated[None, Field(default=None, frozen=True)] = None


class ExponentialDistanceDependentDistribution(DistanceDependentDistribution):
    """Standard exponential distance distribution used by SSCX and thalamus EMC files."""

    name: Annotated[str, Field(default="exp", frozen=True)] = "exp"
    function: Annotated[
        str,
        Field(
            default="(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}",
            frozen=True,
        ),
    ] = "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}"


class StepDistanceDependentDistribution(DistanceDependentDistribution):
    """Step distance distribution used by detailed SSCX models.

    ``{step_begin}`` and ``{step_end}`` are not user-declared placeholders.
    BluePyEModel's ``define_distributions()`` special-cases the name ``step`` and
    computes both values from the imported morphology's calcium hot-spot via
    ``get_hotspot_location()`` (Larkum & Zhu, 2002). Do not add them to
    ``parameters``; they must remain in the function string verbatim.
    """

    name: Annotated[str, Field(default="step", frozen=True)] = "step"
    function: Annotated[
        str,
        Field(
            default="{value} * (0.1 + 0.9 * int(({distance} > {step_begin}) & "
            "({distance} < {step_end})))",
            frozen=True,
        ),
    ] = (
        "{value} * (0.1 + 0.9 * int(({distance} > {step_begin}) & " "({distance} < {step_end})))"
    )


class ExponentialNaDendDistanceDependentDistribution(DistanceDependentDistribution):
    """Exponential dendritic sodium distance distribution used by hippocampus models."""

    name: Annotated[str, Field(default="exp_na_dend", frozen=True)] = "exp_na_dend"
    function: Annotated[
        str,
        Field(default="math.exp((-{distance})/50)*{value}", frozen=True),
    ] = "math.exp((-{distance})/50)*{value}"


class LinearHDApicDistanceDependentDistribution(DistanceDependentDistribution):
    """Linear hot-spot apical distance distribution."""

    name: Annotated[str, Field(default="linear_hd_apic", frozen=True)] = "linear_hd_apic"
    function: Annotated[
        str,
        Field(default="(1. + 3./100. * {distance})*{value}", frozen=True),
    ] = "(1. + 3./100. * {distance})*{value}"


class SigmoidKADApicDistanceDependentDistribution(DistanceDependentDistribution):
    """Sigmoid KAD apical distance distribution."""

    name: Annotated[str, Field(default="sigmoid_kad_apic", frozen=True)] = "sigmoid_kad_apic"
    function: Annotated[
        str,
        Field(
            default="(15./(1. + math.exp((300-{distance})/50)))*{value}",
            frozen=True,
        ),
    ] = "(15./(1. + math.exp((300-{distance})/50)))*{value}"


class LinearEPasApicDistanceDependentDistribution(DistanceDependentDistribution):
    """Linear e_pas apical distance distribution."""

    name: Annotated[str, Field(default="linear_e_pas_apic", frozen=True)] = "linear_e_pas_apic"
    function: Annotated[
        str,
        Field(default="({value}-5*{distance}/150)", frozen=True),
    ] = "({value}-5*{distance}/150)"


class LinearHDPasDistanceDependentDistribution(DistanceDependentDistribution):
    """Linear hot-spot pas distance distribution."""

    name: Annotated[str, Field(default="linear_hdpas", frozen=True)] = "linear_hdpas"
    function: Annotated[
        str,
        Field(default="(1. + 3./100. * {distance})*{value}", frozen=True),
    ] = "(1. + 3./100. * {distance})*{value}"


class SigmoidKADDistanceDependentDistribution(DistanceDependentDistribution):
    """Sigmoid KAD distance distribution."""

    name: Annotated[str, Field(default="sigmoid_kad", frozen=True)] = "sigmoid_kad"
    function: Annotated[
        str,
        Field(
            default="(15./(1. + math.exp((150-{distance})/10)))*{value}",
            frozen=True,
        ),
    ] = "(15./(1. + math.exp((150-{distance})/10)))*{value}"


class SigmoidKDBMApicDistanceDependentDistribution(DistanceDependentDistribution):
    """Sigmoid KDBM apical distance distribution."""

    name: Annotated[str, Field(default="sigmoid_kdbm_apic", frozen=True)] = "sigmoid_kdbm_apic"
    function: Annotated[
        str,
        Field(
            default="(15./(1. + math.exp(({distance}-50)/50)))*{value}",
            frozen=True,
        ),
    ] = "(15./(1. + math.exp(({distance}-50)/50)))*{value}"


class CustomDistanceDependentDistribution(DistanceDependentDistribution):
    """User-defined distance-dependent distribution for the optimization workflow."""

    function: Annotated[
        str,
        Field(
            min_length=1,
            title="Custom distance function",
            description="Python expression containing at least {value} and {distance}.",
        ),
    ]


DistanceDependentDistributionUnion = Annotated[
    UniformDistanceDependentDistribution
    | ExponentialDistanceDependentDistribution
    | StepDistanceDependentDistribution
    | ExponentialNaDendDistanceDependentDistribution
    | LinearHDApicDistanceDependentDistribution
    | SigmoidKADApicDistanceDependentDistribution
    | LinearEPasApicDistanceDependentDistribution
    | LinearHDPasDistanceDependentDistribution
    | SigmoidKADDistanceDependentDistribution
    | SigmoidKDBMApicDistanceDependentDistribution
    | CustomDistanceDependentDistribution,
    Discriminator("name"),
]


STANDARD_DISTANCE_DEPENDENT_DISTRIBUTIONS: dict[str, DistanceDependentDistributionUnion] = {
    "uniform": UniformDistanceDependentDistribution(),
    "exp": ExponentialDistanceDependentDistribution(),
    "step": StepDistanceDependentDistribution(),
    "exp_na_dend": ExponentialNaDendDistanceDependentDistribution(),
    "linear_hd_apic": LinearHDApicDistanceDependentDistribution(),
    "sigmoid_kad_apic": SigmoidKADApicDistanceDependentDistribution(),
    "linear_e_pas_apic": LinearEPasApicDistanceDependentDistribution(),
    "linear_hdpas": LinearHDPasDistanceDependentDistribution(),
    "sigmoid_kad": SigmoidKADDistanceDependentDistribution(),
    "sigmoid_kdbm_apic": SigmoidKDBMApicDistanceDependentDistribution(),
}


class IonChannelModelRef(BaseModel):
    """Local reference to an ion channel model by entity ID."""

    model_config = ConfigDict(extra="ignore")

    id_str: str


class OptimizationValueMode(StrEnum):
    """Whether a parameter value is fixed or bounded for optimization."""

    fixed = auto()
    bounds = auto()


class OptimizationValue(BaseModel):
    """A fixed value or an optimizable lower/upper bound pair."""

    model_config = ConfigDict(extra="forbid")

    mode: Annotated[
        OptimizationValueMode,
        Field(default=OptimizationValueMode.fixed),
    ] = OptimizationValueMode.fixed
    value: Annotated[float | None, Field(default=None)] = None
    bounds: Annotated[tuple[float, float] | None, Field(default=None)] = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        """Keep fixed values and bounds mutually exclusive and finite."""
        if self.value is not None and not math.isfinite(self.value):
            msg = "Optimization values must be finite."
            raise ValueError(msg)
        if self.bounds is not None:
            if any(not math.isfinite(bound) for bound in self.bounds):
                msg = "Optimization bounds must be finite."
                raise ValueError(msg)
            if self.bounds[0] > self.bounds[1]:
                msg = "Optimization lower bound must not exceed the upper bound."
                raise ValueError(msg)
        if self.mode == OptimizationValueMode.fixed:
            if self.value is None:
                msg = "A fixed optimization value is required when mode is 'fixed'."
                raise ValueError(msg)
            if self.bounds is not None:
                msg = "Bounds cannot be provided when mode is 'fixed'."
                raise ValueError(msg)
        elif self.value is not None:
            msg = "A fixed value cannot be provided when mode is 'bounds'."
            raise ValueError(msg)
        return self


class ParameterSelection(BaseModel):
    """Value and distance-distribution selection for one regional parameter."""

    model_config = ConfigDict(extra="forbid")

    value: OptimizationValue
    distribution: Annotated[
        str,
        Field(
            default=StandardDistanceDependentDistributionName.uniform,
            min_length=1,
        ),
    ] = StandardDistanceDependentDistributionName.uniform


class GlobalParameterSelection(BaseModel):
    """Value selection for a global parameter."""

    model_config = ConfigDict(extra="forbid")

    value: OptimizationValue
    ion_channel_model: IonChannelModelRef | None = None


class MechanismRegionSelection(BaseModel):
    """One IonChannelModel assigned to a morphology region."""

    model_config = ConfigDict(extra="forbid")

    ion_channel_model: IonChannelModelRef
    parameters: Annotated[
        dict[str, ParameterSelection],
        Field(default_factory=dict),
    ]


def _fixed_parameter(value: float) -> ParameterSelection:
    return ParameterSelection(value=OptimizationValue(value=value))


def _bounded_parameter(lower: float, upper: float) -> ParameterSelection:
    return ParameterSelection(
        value=OptimizationValue(mode=OptimizationValueMode.bounds, bounds=[lower, upper]),
    )


def _default_global_parameters() -> dict[str, GlobalParameterSelection]:
    return {
        "v_init": GlobalParameterSelection(value=OptimizationValue(value=-80.0)),
        "celsius": GlobalParameterSelection(value=OptimizationValue(value=34.0)),
    }


def _default_base_parameters() -> dict[SectionListName, dict[str, ParameterSelection]]:
    return {
        SectionListName.all: {
            "Ra": _fixed_parameter(100.0),
            "g_pas": _bounded_parameter(1e-5, 6e-5),
            "e_pas": _bounded_parameter(-95.0, -60.0),
        },
        SectionListName.myelinated: {"cm": _fixed_parameter(0.02)},
        SectionListName.axonal: {
            "cm": _fixed_parameter(1.0),
            "ena": _fixed_parameter(50.0),
            "ek": _fixed_parameter(-90.0),
        },
        SectionListName.somatic: {
            "cm": _fixed_parameter(1.0),
            "ena": _fixed_parameter(50.0),
            "ek": _fixed_parameter(-90.0),
        },
        SectionListName.apical: {
            "cm": _fixed_parameter(2.0),
            "ena": _fixed_parameter(50.0),
            "ek": _fixed_parameter(-90.0),
        },
        SectionListName.basal: {
            "cm": _fixed_parameter(2.0),
            "ena": _fixed_parameter(50.0),
            "ek": _fixed_parameter(-90.0),
        },
    }


class ParametersSelection(BaseModel):
    """Mechanisms and values selected for the optimization params compiler."""

    model_config = ConfigDict(extra="forbid")

    ion_channel_models: Annotated[
        tuple[IonChannelModelRef, ...],
        Field(default_factory=tuple),
    ]
    mechanism_regions: Annotated[
        dict[SectionListName, tuple[MechanismRegionSelection, ...]],
        Field(default_factory=dict),
    ]
    global_parameters: Annotated[
        dict[str, GlobalParameterSelection],
        Field(default_factory=_default_global_parameters),
    ]
    base_parameters: Annotated[
        dict[SectionListName, dict[str, ParameterSelection]],
        Field(default_factory=_default_base_parameters),
    ]
    distribution_parameters: Annotated[
        dict[str, dict[str, OptimizationValue]],
        Field(default_factory=dict),
    ]

    @model_validator(mode="after")
    def validate_selection_references(self) -> Self:
        """Validate locations and ensure regional references were selected."""
        selected_ids = {model.id_str for model in self.ion_channel_models}
        if len(selected_ids) != len(self.ion_channel_models):
            msg = "ion_channel_models must not contain duplicate entity IDs."
            raise ValueError(msg)
        for location, selections in self.mechanism_regions.items():
            if location not in REGIONAL_PARAMETER_LOCATIONS:
                msg = f"Unsupported mechanism region: {location}."
                raise ValueError(msg)
            for selection in selections:
                if selection.ion_channel_model.id_str not in selected_ids:
                    msg = "Every mechanism region entity must also be listed in ion_channel_models."
                    raise ValueError(msg)
        for name, selection in self.global_parameters.items():
            if (
                selection.ion_channel_model is not None
                and selection.ion_channel_model.id_str not in selected_ids
            ):
                msg = f"Global parameter '{name}' source must also be listed in ion_channel_models."
                raise ValueError(msg)
        for location in self.base_parameters:
            if location not in REGIONAL_PARAMETER_LOCATIONS:
                msg = f"Unsupported base parameter region: {location}."
                raise ValueError(msg)
        return self


class MorphologyCapabilities(BaseModel):
    """Capabilities discovered from the imported source morphology and modifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    has_myelinated: Annotated[bool | None, Field(default=None)] = None
    axonal_section_count: Annotated[NonNegativeInt | None, Field(default=None)] = None
    available_physical_sections: Annotated[
        tuple[PhysicalSectionListName, ...],
        Field(default=()),
    ] = ()
    """Physical section lists (excluding ``myelinated``) with at least one source section.

    Empty by default, which means "not inspected" — callers that construct this model
    directly (e.g. in tests) opt out of the per-region availability check in the compiler.
    Only :func:`preflight_morphology` populates this from a real morphology.
    """


class VariableType(StrEnum):
    """NMODL variable scope for ion-channel parameters."""

    range = "RANGE"
    global_ = "GLOBAL"


@dataclass(frozen=True)
class IonChannelVariable:
    """Normalized variable metadata from an IonChannelModel neuron block."""

    name: str
    source_name: str
    units: str | None
    variable_type: VariableType


@dataclass(frozen=True)
class NormalizedIonChannelModel:
    """Entity metadata required to compile a mechanism and its parameters."""

    entity_id: str
    name: str
    nmodl_suffix: str
    is_stochastic: bool
    is_ljp_corrected: bool
    temperature_celsius: int | None
    range_variables: tuple[IonChannelVariable, ...]
    global_variables: tuple[IonChannelVariable, ...]
    ion_names: frozenset[str] = frozenset()

    @property
    def variables(self) -> tuple[IonChannelVariable, ...]:
        """All RANGE and GLOBAL variables in metadata order."""
        return self.range_variables + self.global_variables

    def find_variable(self, name: str) -> IonChannelVariable | None:
        """Find a variable by its qualified or source NMODL name."""
        qualified_name = name
        if not name.endswith(f"_{self.nmodl_suffix}"):
            qualified_name = f"{name}_{self.nmodl_suffix}"
        for variable in self.variables:
            if variable.name in {name, qualified_name} or variable.source_name == name:
                return variable
        return None


class ParamsDefinitionInput(BaseModel):
    """Local inputs required to compile a BluePyEModel ``params.json`` file."""

    model_config = ConfigDict(extra="forbid")

    parameters_selection: ParametersSelection
    distance_dependent_distributions: Annotated[
        dict[str, DistanceDependentDistribution],
        Field(default_factory=dict),
    ]
    """Distributions declared alongside the built-in standard catalogue.

    Normally user-defined, but a standard name may be declared to override the
    built-in definition, so the base class is accepted here.
    """


class OptimizationArtifactInput(BaseModel):
    """Local inputs required to compile the params/recipe artifact bundle."""

    model_config = ConfigDict(extra="forbid")

    config_contract_version: str
    emodel: str
    parameters_selection: ParametersSelection
    pipeline_settings_overrides: dict[str, Any]
    mtype: str | None
    morphology_filename: str
    distance_dependent_distributions: Annotated[
        dict[str, DistanceDependentDistribution],
        Field(default_factory=dict),
    ]
    """Distributions declared alongside the built-in standard catalogue.

    Normally user-defined, but a standard name may be declared to override the
    built-in definition, so the base class is accepted here.
    """
    morphology_capabilities: MorphologyCapabilities | None = None


PARAMS_ARTIFACT_PATH = "config/params/params.json"
RECIPES_ARTIFACT_PATH = "config/recipes.json"


@dataclass(frozen=True, slots=True)
class OptimizationArtifacts:
    """JSON-ready optimization artifacts and their portable output paths."""

    config_contract_version: str
    artifact_contract_version: str
    params: dict[str, Any]
    recipes: dict[str, dict[str, Any]]
    params_path: str = PARAMS_ARTIFACT_PATH
    recipes_path: str = RECIPES_ARTIFACT_PATH

    def write(self, output_dir: Path) -> None:
        """Write the bundle below an output directory using only contract paths."""
        write_json(output_dir / self.params_path, self.params)
        write_json(output_dir / self.recipes_path, self.recipes)
