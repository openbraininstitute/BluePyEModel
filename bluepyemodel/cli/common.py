"""Options and helpers shared by the CLI subcommands."""

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

import contextlib
import functools
from pathlib import Path

import click

METADATA_OPTIONS = ("etype", "ttype", "mtype", "species", "brain_region", "iteration_tag")


def metadata_options(command):
    """Add the options for the e-model metadata (e-type, species, brain region, ...).

    The metadata is part of the names of the checkpoints and of the entries of final.json, so
    every step of a run must be given the same values.
    """
    options = [
        click.option("--etype", default=None, help="E-type of the e-model."),
        click.option("--ttype", default=None, help="T-type of the e-model."),
        click.option("--mtype", default=None, help="M-type of the e-model."),
        click.option("--species", default=None, help="Species of the e-model."),
        click.option("--brain-region", default=None, help="Brain region of the e-model."),
        click.option(
            "--iteration-tag",
            default=None,
            help="Tag of the run. With a tag, the run uses the directory ./run/<tag>/.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def emodel_options(command):
    """Add the options that identify the e-model: name, recipes file and metadata."""
    command = metadata_options(command)
    options = [
        click.option("--emodel", required=True, help="E-model name, as in the recipes file."),
        click.option(
            "--recipes-path",
            required=True,
            type=click.Path(path_type=Path),
            help="Path to the recipes file.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def parallel_options(command):
    """Add the options that choose how simulations are parallelised."""
    options = [
        click.option(
            "--backend",
            type=click.Choice(["multiprocessing", "ipyparallel", "serial"]),
            default="multiprocessing",
            show_default=True,
            help="How to parallelise the simulations.",
        ),
        click.option(
            "--workers",
            type=int,
            default=None,
            help="Number of processes for the multiprocessing backend. Default: all CPUs.",
        ),
        click.option(
            "--ipyparallel-profile",
            default=None,
            help="ipyparallel profile for the ipyparallel backend.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def given_metadata(**metadata):
    """Return the metadata options that were given on the command line."""
    return {key: value for key, value in metadata.items() if value is not None}


def build_pipeline(emodel, recipes_path, **metadata):
    """Create the EModel_pipeline used by the subcommands, with the local access point."""
    from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline

    return EModel_pipeline(
        emodel=emodel,
        recipes_path=str(recipes_path),
        data_access_point="local",
        **given_metadata(**metadata),
    )


@contextlib.contextmanager
def mapper(backend, workers=None, ipyparallel_profile=None):
    """Yield a map function for the chosen parallel backend."""
    if backend == "serial":
        yield map
    elif backend == "ipyparallel":
        from bluepyemodel.tools.multiprocessing import ipyparallel_map_function

        yield ipyparallel_map_function(profile=ipyparallel_profile)
    else:
        from bluepyemodel.tools.multiprocessing import NestedPool

        with NestedPool(processes=workers) as pool:
            yield pool.map


def pipeline_command(command):
    """Build the pipeline and the map function from the shared options.

    The wrapped function receives ``pipeline`` (an EModel_pipeline whose mapper is the chosen
    map function) instead of the shared options.
    """

    @functools.wraps(command)
    def wrapper(emodel, recipes_path, backend, workers, ipyparallel_profile, **kwargs):
        metadata = {key: kwargs.pop(key) for key in METADATA_OPTIONS}
        pipeline = build_pipeline(emodel, recipes_path, **metadata)
        with mapper(backend, workers, ipyparallel_profile) as map_function:
            pipeline.mapper = map_function
            return command(pipeline=pipeline, **kwargs)

    return wrapper
