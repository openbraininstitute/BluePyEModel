"""Run the whole pipeline: extract, optimise, store, validate, plot and export.

E-features are extracted from the NWB recording in tests/test_data and used as targets for
the single-compartment simplecell model of examples/simplecell. The budget is tiny, so the
test checks that every step works and passes its results to the next, not model quality.
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
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest

from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata
from tests.utils import DATA
from tests.utils import cwd

matplotlib.use("Agg")

SIMPLECELL = Path(__file__).resolve().parents[2] / "examples" / "simplecell"
RECORDING = DATA / "ephys_data" / "C010600C1-MT-C1.nwb"
EMODEL = "simplecell"
SEED = 1
MODEL_DIR = f"emodel={EMODEL}__etype=cADpyr__species=rat__brain_region=SSCX__seed={SEED}"

# IDRest step amplitudes (nA) present in the recording. The middle one is held out for
# validation. The tolerance keeps neighbouring amplitudes (0.053, 0.078 nA) out.
OPTIMISATION_AMPLITUDES = (0.037, 0.075)
VALIDATION_AMPLITUDE = 0.05
EFEATURES = ("Spikecount", "mean_frequency", "voltage_base")


def protocol_name(amplitude):
    return f"IDRest_{amplitude}"


@pytest.fixture
def pipeline_dir(tmp_path):
    """Set up the simplecell model with extraction targets on the NWB recording."""
    dst = tmp_path / "simplecell"
    shutil.copytree(SIMPLECELL / "config", dst / "config")
    shutil.copytree(SIMPLECELL / "morphologies", dst / "morphologies")
    (dst / "ephys_data").mkdir()
    shutil.copyfile(RECORDING, dst / "ephys_data" / RECORDING.name)

    amplitudes = sorted(OPTIMISATION_AMPLITUDES + (VALIDATION_AMPLITUDE,))
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
            for amplitude in amplitudes
            for efeature in EFEATURES
        ],
        "protocols_rheobase": [],
    }
    (dst / "config" / "extract_config" / "simplecell_config.json").write_text(
        json.dumps(extract_config, indent=4)
    )

    recipes_path = dst / "config" / "recipes.json"
    recipes = json.loads(recipes_path.read_text())
    # write the extracted features to a new file, so the test cannot pass by reading the
    # example's features file
    recipes[EMODEL]["features"] = "config/features/extracted.json"
    recipes[EMODEL]["pipeline_settings"].update(
        {
            "extract_absolute_amplitudes": True,
            "plot_extraction": False,
            "validation_protocols": [protocol_name(VALIDATION_AMPLITUDE)],
            "max_ngen": 2,
            "optimisation_params": {"offspring_size": 4},
        }
    )
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


def test_pipeline_end_to_end(pipeline_dir):
    pipeline = EModel_pipeline(
        emodel=EMODEL,
        etype="cADpyr",
        species="rat",
        brain_region="SSCX",
        recipes_path="./config/recipes.json",
        data_access_point="local",
    )

    # extraction
    fitness_configuration = pipeline.extract_efeatures()

    assert (pipeline_dir / "config" / "features" / "extracted.json").is_file()
    protocols = {p.name: p.validation for p in fitness_configuration.protocols}
    assert protocols == {
        protocol_name(OPTIMISATION_AMPLITUDES[0]): False,
        protocol_name(VALIDATION_AMPLITUDE): True,
        protocol_name(OPTIMISATION_AMPLITUDES[1]): False,
    }
    efeatures = {f.name: f.mean for f in fitness_configuration.efeatures}
    assert len(efeatures) == 3 * len(EFEATURES)
    # values of this recording, so that a broken extraction fails the test
    first = protocol_name(OPTIMISATION_AMPLITUDES[0])
    assert efeatures[f"{first}.soma.v.Spikecount"] == pytest.approx(12.0)
    assert efeatures[f"{first}.soma.v.voltage_base"] == pytest.approx(-72.5, abs=0.5)
    assert (
        efeatures[f"{protocol_name(OPTIMISATION_AMPLITUDES[1])}.soma.v.Spikecount"]
        > efeatures[f"{first}.soma.v.Spikecount"]
    )

    # optimisation, storage, validation, plotting, export
    pipeline.optimise(seed=SEED)
    pipeline.store_optimisation_results()
    pipeline.validation()
    pipeline.plot(only_validated=False)
    export_emodels_hoc(pipeline.access_point, only_validated=False, seeds=[SEED])
    export_emodels_sonata(pipeline.access_point, only_validated=False, seeds=[SEED])

    # the stored model was optimised on the extracted targets and validated on the
    # held-out protocol
    final = json.loads((pipeline_dir / "final.json").read_text())
    entry = final[f"{EMODEL}__{SEED}"]
    assert entry["seed"] == SEED
    assert set(entry["parameters"]) == {"gnabar_hh.somatic", "gkbar_hh.somatic"}
    # e-feature names look like "IDRest_0.037.soma.v.Spikecount"
    optimisation_protocols = {name.split(".soma.")[0] for name in entry["fitness"]}
    assert optimisation_protocols == {protocol_name(a) for a in OPTIMISATION_AMPLITUDES}
    validation_protocols = {name.split(".soma.")[0] for name in entry["validation_fitness"]}
    assert validation_protocols == {protocol_name(VALIDATION_AMPLITUDE)}
    assert isinstance(entry["validated"], bool)

    figures = pipeline_dir / "figures" / EMODEL
    for kind in ("optimisation", "traces", "scores"):
        assert list((figures / kind).rglob("*.pdf")), f"no {kind} figure"

    hoc_dir = pipeline_dir / "export_emodels_hoc" / MODEL_DIR
    sonata_dir = pipeline_dir / "export_emodels_sonata" / MODEL_DIR
    assert (sonata_dir / "nodes.h5").is_file()
    assert (sonata_dir / "model.hoc").is_file()

    # the exported hoc model carries the optimised parameters
    conductances = read_exported_conductances(hoc_dir)
    assert conductances["gnabar_hh"] == pytest.approx(entry["parameters"]["gnabar_hh.somatic"])
    assert conductances["gkbar_hh"] == pytest.approx(entry["parameters"]["gkbar_hh.somatic"])
