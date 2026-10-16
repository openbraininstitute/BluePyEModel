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

import pickle

import numpy as np
import pytest

from bluepyemodel.tools import checkpoint_hdf5


class _FakeFitness:
    """Minimal stand-in for a DEAP fitness object."""

    def __init__(self, values, weights=(-1.0, -1.0)):
        self.values = tuple(values)
        self.weights = tuple(weights)


class _FakeIndividual(list):
    """Minimal stand-in for a DEAP individual."""

    def __init__(self, genes, fitness_values):
        super().__init__(genes)
        self.fitness = _FakeFitness(fitness_values)


class _FakeLogbook:
    """Minimal stand-in for a DEAP Logbook."""

    header = ["gen", "nevals", "avg"]

    def __init__(self, data):
        self._data = data

    def select(self, field):
        return self._data.get(field, [])


class _FakeHistory:
    """Minimal stand-in for a DEAP History."""

    def __init__(self, genealogy_history):
        self.genealogy_history = genealogy_history


class _FakeCMA:
    """Minimal stand-in for a CMA_SO/CMA_MO status object."""

    def __init__(self, active):
        self.active = active


@pytest.fixture
def population():
    return [
        _FakeIndividual([0.1, 0.2], [1.0, 2.0]),
        _FakeIndividual([0.3, 0.4], [1.5, 2.5]),
    ]


@pytest.fixture
def ibea_checkpoint(population):
    return {
        "generation": 5,
        "param_names": ["gNaTgbar", "gK_Tstbar"],
        "halloffame": population[:1],
        "population": population,
        "logbook": _FakeLogbook({"gen": [0, 1], "nevals": [10, 10], "avg": [1.2, 1.1]}),
        "history": _FakeHistory({1: [0.1, 0.2], 2: [0.3, 0.4]}),
    }


@pytest.fixture
def cma_checkpoint(population):
    return {
        "generation": 3,
        "param_names": ["gNaTgbar", "gK_Tstbar"],
        "halloffame": population[:1],
        "population": population,
        "logbook": None,
        "history": None,
        "CMA_es": _FakeCMA(active=True),
    }


def _write_pickle_checkpoint(path, checkpoint, seed):
    path = path.with_name(f"emodel=Test__seed={seed}.pkl")
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    return path


def test_detect_optimizer_cma_so():
    class CMA_SO:
        pass

    assert checkpoint_hdf5.detect_optimizer({"CMA_es": CMA_SO()}) == "CMA_SO"


def test_detect_optimizer_cma_mo():
    class CMA_MO:
        pass

    assert checkpoint_hdf5.detect_optimizer({"CMA_es": CMA_MO()}) == "CMA_MO"


def test_detect_optimizer_ibea_with_parents():
    assert checkpoint_hdf5.detect_optimizer({"parents": []}) == "IBEA"


def test_detect_optimizer_defaults_to_ibea():
    assert checkpoint_hdf5.detect_optimizer({}) == "IBEA"


def test_convert_and_read_roundtrip_ibea(tmp_path, ibea_checkpoint):
    pickle_path = _write_pickle_checkpoint(tmp_path, ibea_checkpoint, seed=42)

    h5_path = checkpoint_hdf5.convert_checkpoint(pickle_path)

    run, seed = checkpoint_hdf5.read_checkpoint_h5(h5_path)

    assert seed == 42
    assert run["generation"] == 5
    assert run["param_names"] == ["gNaTgbar", "gK_Tstbar"]
    assert len(run["population"]) == 2
    assert len(run["halloffame"]) == 1
    assert list(run["population"][0]) == pytest.approx([0.1, 0.2])
    assert run["population"][0].fitness.values == pytest.approx((1.0, 2.0))
    assert run["logbook"].select("gen") == [0, 1]
    assert run["logbook"].header == ["gen", "nevals", "avg"]
    assert run["history"].genealogy_history[1] == pytest.approx([0.1, 0.2])
    assert "CMA_es" not in run


def test_convert_and_read_roundtrip_cma(tmp_path, cma_checkpoint):
    pickle_path = _write_pickle_checkpoint(tmp_path, cma_checkpoint, seed=7)

    h5_path = checkpoint_hdf5.convert_checkpoint(pickle_path)

    run, seed = checkpoint_hdf5.read_checkpoint_h5(h5_path)

    assert seed == 7
    assert "CMA_es" in run
    assert run["CMA_es"].active is True
    run["CMA_es"].check_termination(1)  # no-op, must not raise
    assert run["logbook"] is None
    assert run["history"].genealogy_history == {}


def test_convert_checkpoint_with_explicit_output_path(tmp_path, ibea_checkpoint):
    pickle_path = _write_pickle_checkpoint(tmp_path, ibea_checkpoint, seed=1)
    output_path = tmp_path / "custom_name.h5"

    result = checkpoint_hdf5.convert_checkpoint(pickle_path, output_path=output_path)

    assert result == str(output_path)
    assert output_path.is_file()


def test_convert_checkpoint_with_optimizer_override(tmp_path, ibea_checkpoint):
    pickle_path = _write_pickle_checkpoint(tmp_path, ibea_checkpoint, seed=1)

    h5_path = checkpoint_hdf5.convert_checkpoint(pickle_path, optimizer_override="CMA_SO")

    run, _ = checkpoint_hdf5.read_checkpoint_h5(h5_path)
    assert "CMA_es" in run


def test_convert_checkpoint_with_empty_halloffame_and_population(tmp_path):
    checkpoint = {
        "generation": 0,
        "param_names": [],
        "halloffame": [],
        "population": [],
        "logbook": None,
        "history": None,
    }
    pickle_path = _write_pickle_checkpoint(tmp_path, checkpoint, seed=1)

    h5_path = checkpoint_hdf5.convert_checkpoint(pickle_path)

    run, _ = checkpoint_hdf5.read_checkpoint_h5(h5_path)
    assert run["halloffame"] == []
    assert run["population"] == []


def test_extract_seed_from_path_found():
    assert checkpoint_hdf5._extract_seed_from_path("emodel=L5PC__seed=42.pkl") == 42


def test_extract_seed_from_path_not_found():
    assert checkpoint_hdf5._extract_seed_from_path("no_seed_here.pkl") == -1


def test_individuals_to_arrays_empty():
    genes, fitness_values, fitness_reduce = checkpoint_hdf5._individuals_to_arrays([])
    assert genes.shape == (0, 0)
    assert fitness_values.shape == (0, 0)
    assert fitness_reduce.shape == (0,)


def test_individuals_to_arrays_non_empty(population):
    genes, fitness_values, fitness_reduce = checkpoint_hdf5._individuals_to_arrays(population)
    assert genes.shape == (2, 2)
    assert fitness_values.shape == (2, 2)
    assert fitness_reduce[0] == pytest.approx(3.0)


def test_fitness_shim_properties():
    fitness = checkpoint_hdf5._Fitness(values=[1.0, 2.0])
    assert fitness.valid is True
    assert fitness.reduce == pytest.approx(3.0)
    assert fitness.weighted_reduce == pytest.approx(-3.0)


def test_fitness_shim_empty_values_is_invalid():
    fitness = checkpoint_hdf5._Fitness(values=[])
    assert fitness.valid is False


def test_history_read_from_group_empty(tmp_path):
    import h5py

    h5_path = tmp_path / "empty_history.h5"
    with h5py.File(h5_path, "w") as h5:
        group = h5.create_group("history")
        group.create_dataset("genealogy_genes", data=np.empty((0, 0), dtype=np.float64))

    with h5py.File(h5_path, "r") as h5:
        history = checkpoint_hdf5._read_history_from_group(h5["history"])

    assert history.genealogy_history == {}
