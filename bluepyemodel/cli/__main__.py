"""Module entry point for ``python -m bluepyemodel.cli``."""

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

import click

from bluepyemodel.cli.optimise import optimise


@click.group()
def main():
    """BluePyEModel command-line interface."""


main.add_command(optimise)


if __name__ == "__main__":
    main()
