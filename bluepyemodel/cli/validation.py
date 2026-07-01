"""Validation CLI subcommand."""

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

from pathlib import Path

import click


@click.command()
@click.option("--emodel", required=True, help="EModel name")
@click.option("--workers", required=False, default=None, help="Number of parallel workers.")
@click.option(
    "--recipes-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to recipes file or directory",
)
@click.option(
    "--preselect-for-validation",
    is_flag=True,
    default=False,
    help="Skip models that have already been validated.",
)
def validate(emodel, workers, recipes_path, preselect_for_validation):
    """Validate stored e-models from final.json."""
    from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
    from bluepyemodel.tools.multiprocessing import NestedPool

    with NestedPool(processes=workers) as pool:
        pipeline = EModel_pipeline(
            emodel=emodel,
            recipes_path=recipes_path,
        )
        pipeline.mapper = pool.map
        pipeline.validation(preselect_for_validation=preselect_for_validation)
