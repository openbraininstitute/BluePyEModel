"""Analysis CLI subcommand."""

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


def _resolve_checkpoint_path(access_point, seed, checkpoint_path, checkpoints_dir):
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    from bluepyemodel.tools.utils import get_checkpoint_path

    return get_checkpoint_path(
        access_point.emodel_metadata,
        seed=seed,
        base_dir=checkpoints_dir,
    )


@click.command()
@click.option("--seed", type=int, required=True, help="Random seed")
@click.option("--emodel", required=True, help="EModel name")
@click.option("--workers", required=False, default=None, help="Number of parallel workers.")
@click.option(
    "--checkpoints-dir",
    required=False,
    type=click.Path(path_type=Path),
    default=Path("./checkpoints"),
    help="Directory containing optimisation checkpoints.",
)
@click.option(
    "--checkpoint-path",
    required=False,
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit path to an optimisation checkpoint (.pkl or .h5).",
)
@click.option(
    "--recipes-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to recipes file or directory",
)
@click.option(
    "--em-json-dir",
    required=False,
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Directory where the EModel JSON file will be written.",
)
def analyse(seed, emodel, workers, checkpoints_dir, checkpoint_path, recipes_path, em_json_dir):
    """Analyse an optimisation checkpoint: store results, plot, export, and write EM JSON."""
    from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
    from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata
    from bluepyemodel.optimisation import store_best_model
    from bluepyemodel.tools.create_em_json import create_em_json
    from bluepyemodel.tools.multiprocessing import NestedPool

    with NestedPool(processes=workers) as pool:
        pipeline = EModel_pipeline(
            emodel=emodel,
            recipes_path=recipes_path,
        )
        pipeline.mapper = pool.map
        resolved_checkpoint = _resolve_checkpoint_path(
            pipeline.access_point,
            seed,
            checkpoint_path,
            checkpoints_dir,
        )

        store_best_model(
            pipeline.access_point,
            seed=seed,
            checkpoint_path=str(resolved_checkpoint),
        )
        pipeline.plot(only_validated=False, seeds=[seed])
        export_emodels_sonata(
            pipeline.access_point,
            only_validated=False,
            only_best=False,
            seeds=[seed],
            map_function=pipeline.mapper,
        )
        create_em_json(
            pipeline.access_point,
            seed=seed,
            map_function=pipeline.mapper,
            output_dir=em_json_dir,
        )
