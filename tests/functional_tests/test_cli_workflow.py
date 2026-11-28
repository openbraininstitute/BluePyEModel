"""Build an e-model from a recording with the command line only.

Runs ``bluepyemodel run`` on the NWB recording in tests/test_data with the single-compartment
simplecell model, then the individual commands on its results. The budget is tiny, so the test
checks that the commands work together, not model quality.
"""

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

import json
import shutil
from pathlib import Path

import matplotlib
import pytest
from click.testing import CliRunner

from bluepyemodel.cli.cli import main
from tests.utils import DATA
from tests.utils import cwd

matplotlib.use("Agg")

SIMPLECELL = Path(__file__).resolve().parents[2] / "examples" / "simplecell"
RECORDING = DATA / "ephys_data" / "C010600C1-MT-C1.nwb"
EMODEL = "simplecell"
METADATA = ["--etype", "cADpyr", "--species", "rat", "--brain-region", "SSCX"]
COMMON = ["--emodel", EMODEL, "--recipes-path", "config/recipes.json", "--backend", "serial"]
MODEL_DIR = "emodel=simplecell__etype=cADpyr__species=rat__brain_region=SSCX__seed=1"


@pytest.fixture
def workdir(tmp_path):
    """The simplecell model, with extraction targets on the NWB recording."""
    shutil.copytree(SIMPLECELL / "config", tmp_path / "config")
    shutil.copytree(SIMPLECELL / "morphologies", tmp_path / "morphologies")
    (tmp_path / "ephys_data").mkdir()
    shutil.copyfile(RECORDING, tmp_path / "ephys_data" / RECORDING.name)

    extract_config = {
        "files": [
            {
                "cell_name": RECORDING.stem,
                "filename": RECORDING.stem,
                "filepath": f"./ephys_data/{RECORDING.name}",
                "ecodes": {"IDRest": {}},
                "other_metadata": {},
            }
        ],
        "targets": [
            {
                "efeature": efeature,
                "protocol": "IDRest",
                "amplitude": amplitude,
                "tolerance": 0.002,
                "efel_settings": {"strict_stiminterval": True},
            }
            for amplitude in (0.037, 0.05, 0.075)
            for efeature in ("Spikecount", "mean_frequency", "voltage_base")
        ],
        "protocols_rheobase": [],
    }
    (tmp_path / "config" / "extract_config" / "simplecell_config.json").write_text(
        json.dumps(extract_config)
    )
    recipes_path = tmp_path / "config" / "recipes.json"
    recipes = json.loads(recipes_path.read_text())
    recipes[EMODEL]["features"] = "config/features/extracted.json"
    recipes[EMODEL]["pipeline_settings"].update(
        {
            "extract_absolute_amplitudes": True,
            "plot_extraction": False,
            "validation_protocols": ["IDRest_0.05"],
            "max_ngen": 2,
            "optimisation_params": {"offspring_size": 4},
        }
    )
    recipes_path.write_text(json.dumps(recipes))

    with cwd(tmp_path):
        yield tmp_path


def invoke(*args):
    result = CliRunner().invoke(main, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def test_build_an_emodel_with_the_cli(workdir):
    result = invoke("run", *COMMON, *METADATA, "--seed", "1", "--export-format", "sonata")

    assert "Extracting e-features" in result.output
    assert (workdir / "config" / "features" / "extracted.json").is_file()
    assert (workdir / "checkpoints" / EMODEL).is_dir()

    final = json.loads((workdir / "final.json").read_text())
    entry = final[f"{EMODEL}__1"]
    assert {name.split(".soma.")[0] for name in entry["fitness"]} == {
        "IDRest_0.037",
        "IDRest_0.075",
    }
    assert {name.split(".soma.")[0] for name in entry["validation_fitness"]} == {"IDRest_0.05"}
    assert list((workdir / "figures" / EMODEL / "traces").rglob("*.pdf"))
    assert (workdir / "export_emodels_sonata" / MODEL_DIR / "nodes.h5").is_file()

    # the steps also run one by one on the same results, found through the metadata
    result = invoke("run", *COMMON, *METADATA, "--no-plot")
    assert "Using the existing features file" in result.output

    invoke("export", *COMMON, *METADATA, "--format", "hoc", "--seed", "1", "--output-dir", "hoc")
    assert (workdir / "hoc" / "model.hoc").is_file()

    result = invoke("summarize", *COMMON, *METADATA)
    assert EMODEL in result.output
