"""Recipe helpers for optimization working-directory staging."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bluepyemodel.preprocessing.schemas import DEFAULT_SECTION_LIST_CATALOG
from bluepyemodel.preprocessing.schemas import PARAMS_ARTIFACT_PATH


def update_pipeline_settings(
    recipes: dict,
    emodel: str,
    overrides: Mapping[str, Any],
) -> dict:
    """Merge ``overrides`` into ``recipes[emodel]['pipeline_settings']``.

    ``None`` values in ``overrides`` are skipped so callers can pass partial
    blocks without clobbering existing recipe settings.
    """
    if emodel not in recipes:
        msg = f"emodel '{emodel}' not in recipes (got keys: {list(recipes)})"
        raise KeyError(msg)

    settings = recipes[emodel].setdefault("pipeline_settings", {})
    for key, value in overrides.items():
        if value is None:
            continue
        settings[key] = value
    return recipes


def _validate_relative_filename(filename: str, field_name: str) -> None:
    path = Path(filename)
    if path.is_absolute() or path.name != filename or ".." in path.parts:
        msg = f"{field_name} must be a relative filename, got {filename!r}."
        raise ValueError(msg)


def build_optimization_recipe(
    emodel: str,
    mtype: str | None,
    morph_filename: str,
    params_filename: str = "params.json",
) -> dict[str, dict[str, Any]]:
    """Build the deterministic BluePyEModel recipe for one optimization run."""
    _validate_relative_filename(morph_filename, "morph_filename")
    _validate_relative_filename(params_filename, "params_filename")
    if params_filename != Path(PARAMS_ARTIFACT_PATH).name:
        msg = f"params_filename must be {Path(PARAMS_ARTIFACT_PATH).name!r}."
        raise ValueError(msg)
    return {
        emodel: {
            "morph_path": "./morphologies/",
            "morphology": [[mtype, morph_filename]],
            "features": f"config/features/{emodel}.json",
            "params": f"config/params/{params_filename}",
            "multiloc_map": DEFAULT_SECTION_LIST_CATALOG.to_recipe_multiloc_map(),
        }
    }
