"""BluePyEModel command-line interface."""

import logging

import click

from bluepyemodel.cli.analysis import analyse
from bluepyemodel.cli.optimisation import optimise
from bluepyemodel.cli.steps import configure_model
from bluepyemodel.cli.steps import export
from bluepyemodel.cli.steps import extract
from bluepyemodel.cli.steps import plot
from bluepyemodel.cli.steps import run
from bluepyemodel.cli.steps import store
from bluepyemodel.cli.steps import summarize
from bluepyemodel.cli.steps import validate

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
    """BluePyEModel command-line interface.

    Build an e-model step by step (extract, optimise, store, validate, plot, export), or all at
    once with ``run``. Every step reads the same recipes file, and takes the same e-model name
    and metadata options, which name the checkpoints and the entries of final.json.
    """
    logging.basicConfig(
        level=LOG_LEVELS[log_level],
        handlers=[logging.StreamHandler()],
        force=True,
    )
    # fontTools logs every font it subsets when matplotlib saves figures
    logging.getLogger("fontTools").setLevel(max(LOG_LEVELS[log_level], logging.WARNING))


main.add_command(extract)
main.add_command(configure_model)
main.add_command(optimise)
main.add_command(store)
main.add_command(validate)
main.add_command(plot)
main.add_command(export)
main.add_command(summarize)
main.add_command(run)
main.add_command(analyse)
