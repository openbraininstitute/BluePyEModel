"""Run the simplecell example through the whole pipeline: optimise, store, validate, plot, export."""

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
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest

from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata
from tests.utils import cwd

matplotlib.use("Agg")

SIMPLECELL = Path(__file__).resolve().parents[2] / "examples" / "simplecell"
EMODEL = "simplecell"
SEED = 1
MODEL_DIR = f"emodel={EMODEL}__etype=cADpyr__species=rat__brain_region=SSCX__seed={SEED}"


@pytest.fixture
def simplecell_dir(tmp_path):
    """Copy the simplecell example, with a tiny optimisation budget, to tmp_path."""
    dst = tmp_path / "simplecell"
    shutil.copytree(SIMPLECELL / "config", dst / "config")
    shutil.copytree(SIMPLECELL / "morphologies", dst / "morphologies")

    recipes_path = dst / "config" / "recipes.json"
    recipes = json.loads(recipes_path.read_text())
    settings = recipes[EMODEL]["pipeline_settings"]
    settings["max_ngen"] = 2
    settings["optimisation_params"] = {"offspring_size": 4}
    recipes_path.write_text(json.dumps(recipes, indent=4))

    with cwd(dst):
        yield dst


def read_exported_conductances(model_dir):
    """Load an exported hoc template in a fresh NEURON and return its soma conductances.

    A fresh process is used because the pipeline has already defined a hoc template with
    the same name in this one, and NEURON cannot redefine templates. It is also how a user
    loads an exported model.
    """
    script = f"""
import json
from neuron import h
h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")
h.load_file({str(model_dir / "model.hoc")!r})
cell = h.{EMODEL}({str(model_dir)!r}, "simple.swc")
seg = cell.soma[0](0.5)
print(json.dumps({{"gnabar_hh": seg.gnabar_hh, "gkbar_hh": seg.gkbar_hh}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_simplecell_end_to_end(simplecell_dir):
    pipeline = EModel_pipeline(
        emodel=EMODEL,
        etype="cADpyr",
        species="rat",
        brain_region="SSCX",
        recipes_path="./config/recipes.json",
        data_access_point="local",
    )

    pipeline.optimise(seed=SEED)
    pipeline.store_optimisation_results()
    pipeline.validation()
    pipeline.plot(only_validated=False)
    export_emodels_hoc(pipeline.access_point, only_validated=False, seeds=[SEED])
    export_emodels_sonata(pipeline.access_point, only_validated=False, seeds=[SEED])

    # storage and validation
    final = json.loads((simplecell_dir / "final.json").read_text())
    entry = final[f"{EMODEL}__{SEED}"]
    assert entry["seed"] == SEED
    assert set(entry["parameters"]) == {"gnabar_hh.somatic", "gkbar_hh.somatic"}
    assert entry["fitness"]
    assert entry["validation_fitness"]
    assert isinstance(entry["validated"], bool)

    # plotting
    figures = simplecell_dir / "figures" / EMODEL
    for kind in ("optimisation", "traces", "scores"):
        assert list((figures / kind).rglob("*.pdf")), f"no {kind} figure"

    # export: both formats, and the hoc model carries the optimised parameters
    hoc_dir = simplecell_dir / "export_emodels_hoc" / MODEL_DIR
    sonata_dir = simplecell_dir / "export_emodels_sonata" / MODEL_DIR
    assert (sonata_dir / "nodes.h5").is_file()
    assert (sonata_dir / "model.hoc").is_file()

    conductances = read_exported_conductances(hoc_dir)
    assert conductances["gnabar_hh"] == pytest.approx(entry["parameters"]["gnabar_hh.somatic"])
    assert conductances["gkbar_hh"] == pytest.approx(entry["parameters"]["gkbar_hh.somatic"])
