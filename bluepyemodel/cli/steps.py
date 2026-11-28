"""CLI subcommands for the steps of the e-model building pipeline."""

"""
Copyright 2025, Open Brain Institute

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

from bluepyemodel.cli.common import emodel_options
from bluepyemodel.cli.common import parallel_options
from bluepyemodel.cli.common import pipeline_command

L = logging.getLogger(__name__)

seed_option = click.option(
    "--seed",
    "seeds",
    type=int,
    multiple=True,
    help="Seed of the optimisation. Can be repeated. Default: every seed found.",
)


def _seeds_or_none(seeds):
    return list(seeds) if seeds else None


def _features_path(pipeline):
    access_point = pipeline.access_point
    return Path(access_point.emodel_dir) / access_point.get_recipes()["features"]


@click.command()
@emodel_options
@parallel_options
@pipeline_command
def extract(pipeline):
    """Extract e-features from the recordings and write the features file of the recipe.

    The recordings and targets are read from the json file given by the pipeline setting
    path_extract_config.
    """
    fitness_configuration = pipeline.extract_efeatures()
    click.echo(
        f"Extracted {len(fitness_configuration.efeatures)} e-features for "
        f"{len(fitness_configuration.protocols)} protocols to {_features_path(pipeline)}"
    )


@click.command("configure-model")
@emodel_options
@parallel_options
@click.option("--morphology-name", required=True, help="Morphology file name, without extension.")
@click.option("--morphology-path", default=None, help="Path to the morphology file.")
@click.option("--morphology-format", type=click.Choice(["asc", "swc"]), default=None)
@click.option(
    "--use-gene-data",
    is_flag=True,
    default=False,
    help="Select the ion channels with the ion channel selector, from the gene expression "
    "table (setting name_gene_map) and the ion channel map (setting path_ic_map).",
)
@pipeline_command
def configure_model(pipeline, morphology_name, morphology_path, morphology_format, use_gene_data):
    """Create a model configuration and write it over the params file of the recipe."""
    configuration = pipeline.configure_model(
        morphology_name=morphology_name,
        morphology_path=morphology_path,
        morphology_format=morphology_format,
        use_gene_data=use_gene_data,
    )
    click.echo(
        f"Model configuration with {len(configuration.mechanisms)} mechanisms and "
        f"{len(configuration.parameters)} parameters written to "
        f"{pipeline.access_point.get_recipes()['params']}"
    )


@click.command()
@emodel_options
@parallel_options
@seed_option
@pipeline_command
def store(pipeline, seeds):
    """Store the best model of each optimisation checkpoint in final.json."""
    if seeds:
        for seed in seeds:
            pipeline.store_optimisation_results(seed=seed)
    else:
        pipeline.store_optimisation_results()


@click.command()
@emodel_options
@parallel_options
@pipeline_command
def validate(pipeline):
    """Run the validation protocols on the stored models and record whether they pass."""
    pipeline.validation()


@click.command()
@emodel_options
@parallel_options
@seed_option
@click.option("--only-validated", is_flag=True, default=False, help="Only plot validated models.")
@pipeline_command
def plot(pipeline, seeds, only_validated):
    """Plot the stored models: optimisation, traces, scores and parameters."""
    pipeline.plot(only_validated=only_validated, seeds=_seeds_or_none(seeds))


@click.command()
@emodel_options
@parallel_options
@seed_option
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["hoc", "sonata"]),
    default="sonata",
    show_default=True,
    help="Export format.",
)
@click.option("--only-validated", is_flag=True, default=False, help="Only export validated models.")
@click.option(
    "--all-models",
    is_flag=True,
    default=False,
    help="Export every selected model instead of only the best one.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write the files to. Only with a single --seed. Default: one directory "
    "per model in ./export_emodels_<format>/.",
)
@pipeline_command
def export(pipeline, seeds, export_format, only_validated, all_models, output_dir):
    """Export stored models to hoc, or to SONATA (hoc, nodes file and morphology)."""
    from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
    from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata

    if output_dir is not None and len(seeds) != 1:
        raise click.UsageError("--output-dir needs exactly one --seed.")

    kwargs = {
        "only_validated": only_validated,
        "only_best": not all_models,
        "seeds": _seeds_or_none(seeds),
        "output_dir": output_dir,
    }
    if export_format == "hoc":
        export_emodels_hoc(pipeline.access_point, **kwargs)
    else:
        export_emodels_sonata(pipeline.access_point, map_function=pipeline.mapper, **kwargs)


@click.command()
@emodel_options
@parallel_options
@pipeline_command
def summarize(pipeline):
    """Print a summary of the e-model building: configuration and stored models."""
    click.echo(str(pipeline.access_point))


@click.command()
@emodel_options
@parallel_options
@click.option(
    "--seed",
    "seeds",
    type=int,
    multiple=True,
    default=(1,),
    show_default=True,
    help="Seed of an optimisation. Can be repeated to run several optimisations.",
)
@click.option(
    "--force-extraction",
    is_flag=True,
    default=False,
    help="Extract the e-features even if the features file of the recipe exists.",
)
@click.option(
    "--export-format",
    type=click.Choice(["hoc", "sonata"]),
    default="sonata",
    show_default=True,
)
@click.option("--no-plot", is_flag=True, default=False, help="Skip the plotting step.")
@pipeline_command
def run(pipeline, seeds, force_extraction, export_format, no_plot):
    """Run the whole pipeline: extract, optimise, store, validate, plot and export.

    Extraction is skipped when the features file of the recipe already exists, unless
    --force-extraction is given. An optimisation whose checkpoint exists continues from its
    last generation.
    """
    from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
    from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata

    features_path = _features_path(pipeline)
    if force_extraction or not features_path.is_file():
        click.echo("Extracting e-features")
        pipeline.extract_efeatures()
    else:
        click.echo(f"Using the existing features file {features_path}")

    for seed in seeds:
        click.echo(f"Optimising seed {seed}")
        pipeline.optimise(seed=seed)
        pipeline.store_optimisation_results(seed=seed)

    click.echo("Validating")
    pipeline.validation()

    if not no_plot:
        click.echo("Plotting")
        pipeline.plot(seeds=list(seeds))

    click.echo(f"Exporting to {export_format}")
    if export_format == "hoc":
        export_emodels_hoc(pipeline.access_point, seeds=list(seeds))
    else:
        export_emodels_sonata(
            pipeline.access_point, seeds=list(seeds), map_function=pipeline.mapper
        )
