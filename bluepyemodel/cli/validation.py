"""Validation CLI subcommand."""

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
