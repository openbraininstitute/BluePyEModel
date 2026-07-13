"""BluePyEModel command-line interface."""

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

import click

from bluepyemodel.cli.analysis import analyse
from bluepyemodel.cli.optimisation import optimise
from bluepyemodel.cli.validation import validate

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(sorted(LOG_LEVELS), case_sensitive=False),
    help="Logging verbosity for BluePyEModel commands.",
)
def main(log_level):
    """BluePyEModel command-line interface."""
    logging.basicConfig(
        level=LOG_LEVELS[log_level],
        handlers=[logging.StreamHandler()],
        force=True,
    )


main.add_command(analyse)
main.add_command(optimise)
main.add_command(validate)
