"""Optimisation CLI subcommand."""

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

import logging
from pathlib import Path

import click


@click.command()
@click.option("--seed", type=int, required=True, help="Random seed")
@click.option("--emodel", required=True, help="EModel name")
@click.option(
    "--recipes-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to recipes file or directory",
)
def optimise(seed, emodel, recipes_path):
    """Run EModel optimisation."""
    from bluepyemodel.access_point.local import LocalAccessPoint
    from bluepyemodel.optimisation import setup_and_run_optimisation
    from bluepyemodel.tools.multiprocessing import NestedPool

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ],
    )
    access_point = LocalAccessPoint(
        emodel=emodel,
        recipes_path=recipes_path,
    )
    with NestedPool() as pool:
        setup_and_run_optimisation(
            access_point,
            seed=seed,
            mapper=pool.map,
            terminator=None,
        )
