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

L = logging.getLogger(__name__)


@click.command()
@click.option("--seed", type=int, required=True, help="Random seed")
@click.option("--emodel", required=True, help="EModel name")
@click.option("--workers", required=False, default=None, help="Number of parallel workers.")
@click.option(
    "--checkpoints-dir",
    required=False,
    type=click.Path(path_type=Path),
    default=Path("./checkpoints"),
)
@click.option(
    "--recipes-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to recipes file or directory",
)
@click.option(
    "--convert-checkpoint",
    is_flag=True,
    default=False,
    help="Convert the optimisation checkpoint pickle to HDF5 after the run.",
)
def optimise(seed, emodel, workers, checkpoints_dir, recipes_path, convert_checkpoint):
    """Run EModel optimisation."""
    from bluepyemodel.access_point.local import LocalAccessPoint
    from bluepyemodel.optimisation import setup_and_run_optimisation
    from bluepyemodel.tools.conversion import pickle_to_hdf5
    from bluepyemodel.tools.multiprocessing import NestedPool
    from bluepyemodel.tools.utils import get_checkpoint_path

    access_point = LocalAccessPoint(
        emodel=emodel,
        recipes_path=recipes_path,
    )
    with NestedPool(processes=workers) as pool:
        L.info("Running optimisation with %n workers", pool._processes)
        setup_and_run_optimisation(
            access_point,
            seed=seed,
            mapper=pool.map,
            terminator=None,
            checkpoints_dir=checkpoints_dir,
        )

    if convert_checkpoint:
        L.info("Convering pickle checkpoint to hdf5...")
        checkpoint_path = Path(
            get_checkpoint_path(
                access_point.emodel_metadata,
                seed=seed,
                base_dir=checkpoints_dir,
            )
        )
        pickle_to_hdf5(checkpoint_path, checkpoint_path.with_suffix(".h5"))
