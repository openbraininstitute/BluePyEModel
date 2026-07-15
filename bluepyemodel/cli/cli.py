"""BluePyEModel command-line interface."""

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
