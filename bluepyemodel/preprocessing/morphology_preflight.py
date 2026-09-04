"""Runtime morphology capability checks for optimization artifact compilation."""

from pathlib import Path
from typing import Any

import morphio

from bluepyemodel.preprocessing.schemas import (
    PHYSICAL_SECTION_LIST_NAMES,
    AxonModifier,
    MorphologyCapabilities,
    PhysicalSectionListName,
)

__all__ = ["MorphologyCapabilities", "load_morphology_nrn_order", "preflight_morphology"]

_SECTION_TYPE_TO_PHYSICAL_NAME: dict[Any, PhysicalSectionListName] = {
    morphio.SectionType.soma: PhysicalSectionListName.somatic,
    morphio.SectionType.apical_dendrite: PhysicalSectionListName.apical,
    morphio.SectionType.basal_dendrite: PhysicalSectionListName.basal,
    morphio.SectionType.axon: PhysicalSectionListName.axonal,
}

_MINIMUM_SOURCE_AXONAL_SECTIONS: dict[AxonModifier, int] = {
    AxonModifier.replace_axon_with_taper: 3,
    AxonModifier.replace_axon_legacy: 2,
}


def load_morphology_nrn_order(path: Path) -> morphio.Morphology:
    """Load morphology with NEURON-compatible section ordering."""
    collection = morphio.Collection(str(path.parent), extensions=[path.suffix])
    return collection.load(path.stem, morphio.Option.nrn_order)


def _count_axonal_sections(morphology: Any) -> int:
    """Count source sections classified as axon by MorphIO."""
    return sum(section.type == morphio.SectionType.axon for section in morphology.sections)


def _available_physical_sections(morphology: Any) -> tuple[PhysicalSectionListName, ...]:
    """Return the physical section lists with at least one source section."""
    present: set[PhysicalSectionListName] = set()
    for section in morphology.sections:
        physical_name = _SECTION_TYPE_TO_PHYSICAL_NAME.get(section.type)
        if physical_name is not None:
            present.add(physical_name)

    # MorphIO keeps soma points on a dedicated ``soma`` object rather than emitting a
    # ``SectionType.soma`` entry in ``sections``, so the section scan above never sees it.
    soma = getattr(morphology, "soma", None)
    soma_points = getattr(soma, "points", None)
    if soma_points is not None and len(soma_points) > 0:
        present.add(PhysicalSectionListName.somatic)

    return tuple(name for name in PHYSICAL_SECTION_LIST_NAMES if name in present)


def preflight_morphology(
    path: Path,
    axon_modifier: AxonModifier | str,
) -> MorphologyCapabilities:
    """Load the staged morphology and return its section-list capabilities.

    A staged SWC/ASC asset cannot establish a populated runtime myelinated section
    list. In no-replacement mode, ``has_myelinated`` therefore remains unknown and
    myelinated parameter rows are rejected.
    """
    if not path.is_file():
        msg = f"Morphology preflight asset does not exist: {path}."
        raise ValueError(msg)

    modifier = AxonModifier(axon_modifier)
    morphology = load_morphology_nrn_order(path)
    axonal_section_count = _count_axonal_sections(morphology)
    minimum = _MINIMUM_SOURCE_AXONAL_SECTIONS.get(modifier)
    if minimum is not None and axonal_section_count < minimum:
        msg = (
            f"Morphology '{path.name}' has {axonal_section_count} source axon sections, "
            f"but axon modifier '{modifier.value}' requires at least {minimum}."
        )
        raise ValueError(msg)

    if modifier in {
        AxonModifier.replace_axon_with_taper,
        AxonModifier.replace_axon_olfactory_bulb,
    }:
        has_myelinated = True
    elif modifier in {
        AxonModifier.replace_axon_legacy,
        AxonModifier.bluepyopt_replace_axon,
    }:
        has_myelinated = False
    else:
        has_myelinated = None

    return MorphologyCapabilities(
        has_myelinated=has_myelinated,
        axonal_section_count=axonal_section_count,
        available_physical_sections=_available_physical_sections(morphology),
    )
