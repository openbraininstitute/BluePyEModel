"""Tests for source-morphology capability preflight."""

import pytest

from bluepyemodel.preprocessing.morphology_preflight import preflight_morphology
from bluepyemodel.preprocessing.schemas import AxonModifier, PhysicalSectionListName

# Three-point soma (type 1) followed by an unbranched axon (type 2) and a basal
# dendrite (type 3). MorphIO stores the soma samples on ``morphology.soma`` and
# leaves them out of ``morphology.sections``.
_SWC = """\
1 1 0.0 0.0 0.0 1.0 -1
2 1 0.0 -1.0 0.0 1.0 1
3 1 0.0 1.0 0.0 1.0 1
4 2 0.0 1.0 0.0 0.5 1
5 2 0.0 5.0 0.0 0.5 4
6 2 0.0 9.0 0.0 0.5 5
7 3 0.0 -1.0 0.0 0.5 1
8 3 0.0 -5.0 0.0 0.5 7
"""


def _write_morphology(tmp_path, content=_SWC):
    path = tmp_path / "morphology.swc"
    path.write_text(content, encoding="utf-8")
    return path


def test_preflight_reports_somatic_from_soma_points(tmp_path):
    capabilities = preflight_morphology(_write_morphology(tmp_path), AxonModifier.none)
    assert PhysicalSectionListName.somatic in capabilities.available_physical_sections


def test_preflight_reports_present_neurite_sections(tmp_path):
    capabilities = preflight_morphology(_write_morphology(tmp_path), AxonModifier.none)
    assert capabilities.available_physical_sections == (
        PhysicalSectionListName.somatic,
        PhysicalSectionListName.basal,
        PhysicalSectionListName.axonal,
    )
    assert capabilities.axonal_section_count == 1


def test_no_replacement_leaves_myelination_unknown(tmp_path):
    capabilities = preflight_morphology(_write_morphology(tmp_path), AxonModifier.none)
    assert capabilities.has_myelinated is None


def test_tapered_replacement_requires_enough_source_axon_sections(tmp_path):
    with pytest.raises(ValueError, match="source axon sections"):
        preflight_morphology(
            _write_morphology(tmp_path, "1 1 0.0 0.0 0.0 1.0 -1\n"),
            AxonModifier.replace_axon_with_taper,
        )


def test_preflight_rejects_a_missing_asset(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        preflight_morphology(tmp_path / "absent.swc", AxonModifier.none)
