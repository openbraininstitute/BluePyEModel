"""Write an EModel resource JSON file from a stored local emodel."""

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
import logging
from pathlib import Path

from bluepyemodel.evaluation.evaluation import compute_responses, get_evaluator_from_access_point

logger = logging.getLogger(__name__)


def create_em_json(access_point, seed, map_function=map, output_dir=None):
    """Create an EModel JSON file for a stored emodel seed.

    The output follows the Nexus EModel resource layout produced by
    :meth:`bluepyemodel.emodel_pipeline.emodel.EModel.as_dict`. Responses
    (including holding and threshold currents) are computed before writing.

    Args:
        access_point (DataAccessPoint): data access point.
        seed (int): optimisation seed of the emodel to export.
        map_function (map): parallel map function used for simulations.
        output_dir (str or Path): directory for the output file. Defaults to
            the current working directory.

    Returns:
        Path: path to the written JSON file.
    """
    cell_evaluator = get_evaluator_from_access_point(
        access_point,
        include_validation_protocols=True,
    )
    emodels = compute_responses(
        access_point,
        cell_evaluator,
        map_function,
        seeds=[seed],
        preselect_for_validation=False,
        store_responses=False,
    )
    if not emodels:
        msg = f"No emodel found for seed {seed}."
        raise ValueError(msg)

    emodel = emodels[0]
    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"EM__{access_point.emodel_metadata.as_string(seed=seed)}.json"

    with output_path.open("w") as json_file:
        json.dump(emodel.as_dict(), json_file, indent=2)

    logger.info("Written EModel JSON to %s", output_path)
    return output_path
