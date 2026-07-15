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

import h5py
import numpy as np
import pytest

from bluepyemodel.tools.conversion import CONTENT_TYPE_CELLS
from bluepyemodel.tools.conversion import CONTENT_TYPE_OPTIMISATION_SUMMARY
from bluepyemodel.tools.conversion import CONTENT_TYPE_PROTOCOLS
from bluepyemodel.tools.conversion import FORMAT_VERSION
from bluepyemodel.tools.conversion import _CMAStatus
from bluepyemodel.tools.conversion import _Fitness
from bluepyemodel.tools.conversion import _GeneOnlyIndividual
from bluepyemodel.tools.conversion import _History
from bluepyemodel.tools.conversion import _Individual
from bluepyemodel.tools.conversion import _Logbook
from bluepyemodel.tools.conversion import checkpoint_hdf5_to_pickle
from bluepyemodel.tools.conversion import checkpoint_pickle_to_hdf5
from bluepyemodel.tools.conversion import hdf5_to_pickle
from bluepyemodel.tools.conversion import load_checkpoint_hdf5
from bluepyemodel.tools.conversion import pickle_to_hdf5


def _sample_checkpoint():
    return {
        "generation": 3,
        "param_names": ["gNa", "gK"],
        "halloffame": [
            _Individual([0.1, 0.2], fitness_values=[1.5], fitness_weights=[-1.0]),
        ],
        "population": [
            _Individual([0.3, 0.4], fitness_values=[2.0], fitness_weights=[-1.0]),
            _Individual([0.5, 0.6], fitness_values=[2.5], fitness_weights=[-1.0]),
        ],
        "logbook": _Logbook({"gen": [1, 2, 3], "nevals": [10, 20, 30]}, ["gen", "nevals"]),
        "history": _History({1: _GeneOnlyIndividual([0.1, 0.2])}),
        "CMA_es": _CMAStatus(active=True),
    }


def test_checkpoint_pickle_hdf5_roundtrip(tmp_path):
    pickle_path = tmp_path / "emodel=L5PC__seed=7.pkl"
    hdf5_path = tmp_path / "emodel=L5PC__seed=7.h5"
    restored_pickle_path = tmp_path / "restored.pkl"

    checkpoint = _sample_checkpoint()
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(checkpoint, pickle_file)

    checkpoint_pickle_to_hdf5(pickle_path, hdf5_path)
    loaded = load_checkpoint_hdf5(hdf5_path)

    assert loaded["generation"] == 3
    assert loaded["param_names"] == ["gNa", "gK"]
    assert len(loaded["halloffame"]) == 1
    assert list(loaded["halloffame"][0]) == [0.1, 0.2]
    assert loaded["halloffame"][0].fitness.values == (1.5,)
    assert len(loaded["population"]) == 2
    assert loaded["logbook"].select("gen") == [1, 2, 3]
    assert loaded["CMA_es"].active is True

    checkpoint_hdf5_to_pickle(hdf5_path, restored_pickle_path)
    with open(restored_pickle_path, "rb") as pickle_file:
        restored = pickle.load(pickle_file, encoding="latin1")

    assert restored["generation"] == 3
    assert restored["param_names"] == ["gNa", "gK"]
    assert len(restored["halloffame"]) == 1
    assert list(restored["halloffame"][0]) == [0.1, 0.2]


def test_pickle_to_hdf5_infers_optimisation_summary(tmp_path):
    pickle_path = tmp_path / "emodel=L5PC__seed=3.pkl"
    hdf5_path = tmp_path / "checkpoint.h5"

    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(_sample_checkpoint(), pickle_file)

    pickle_to_hdf5(pickle_path, hdf5_path)

    with h5py.File(hdf5_path, "r") as hdf5_file:
        assert hdf5_file.attrs["content_type"] == CONTENT_TYPE_OPTIMISATION_SUMMARY
        assert hdf5_file.attrs["format_version"] == FORMAT_VERSION
        assert hdf5_file.attrs["seed"] == 3
        assert np.allclose(hdf5_file.attrs["fitness_weights"], [-1.0])


def test_hdf5_to_pickle_dispatches_optimisation_summary(tmp_path):
    pickle_path = tmp_path / "emodel=L5PC__seed=5.pkl"
    hdf5_path = tmp_path / "checkpoint.h5"
    restored_pickle_path = tmp_path / "restored.pkl"

    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(_sample_checkpoint(), pickle_file)
    pickle_to_hdf5(pickle_path, hdf5_path)

    hdf5_to_pickle(hdf5_path, restored_pickle_path)

    with open(restored_pickle_path, "rb") as pickle_file:
        restored = pickle.load(pickle_file, encoding="latin1")
    assert restored["generation"] == 3


def test_checkpoint_pickle_hdf5_roundtrip_with_logbook_and_history(tmp_path):
    pickle_path = tmp_path / "emodel=L5PC__seed=7.pkl"
    hdf5_path = tmp_path / "emodel=L5PC__seed=7.h5"

    checkpoint = _sample_checkpoint()
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(checkpoint, pickle_file)

    checkpoint_pickle_to_hdf5(pickle_path, hdf5_path)
    loaded = load_checkpoint_hdf5(hdf5_path)

    assert loaded["logbook"].select("gen") == [1, 2, 3]
    assert len(loaded["history"].genealogy_history) == 1


def test_load_checkpoint_hdf5_without_fitness_weights(tmp_path):
    pickle_path = tmp_path / "emodel=L5PC__seed=2.pkl"
    hdf5_path = tmp_path / "checkpoint.h5"
    checkpoint = {
        "generation": 1,
        "param_names": [],
        "halloffame": [],
        "population": [],
    }

    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(checkpoint, pickle_file)
    checkpoint_pickle_to_hdf5(pickle_path, hdf5_path, optimizer_override="IBEA")

    loaded = load_checkpoint_hdf5(hdf5_path)

    assert loaded["generation"] == 1
    assert loaded["halloffame"] == []
    assert loaded["population"] == []


def test_load_checkpoint_hdf5_rejects_unsupported_format(tmp_path):
    hdf5_path = tmp_path / "bad.h5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.attrs["content_type"] = CONTENT_TYPE_OPTIMISATION_SUMMARY
        hdf5_file.attrs["format_version"] = 99

    with pytest.raises(ValueError, match="Unsupported HDF5 conversion format version"):
        load_checkpoint_hdf5(hdf5_path)


def test_pickle_to_hdf5_rejects_unknown_content(tmp_path):
    pickle_path = tmp_path / "unknown.pkl"
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump({"foo": "bar"}, pickle_file)

    with pytest.raises(ValueError, match="Could not infer pickle content type"):
        pickle_to_hdf5(pickle_path, tmp_path / "out.h5")


def test_checkpoint_pickle_to_hdf5_uses_optimizer_override(tmp_path):
    pickle_path = tmp_path / "emodel=L5PC__seed=1.pkl"
    hdf5_path = tmp_path / "checkpoint.h5"
    checkpoint = _sample_checkpoint()
    del checkpoint["CMA_es"]

    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(checkpoint, pickle_file)

    checkpoint_pickle_to_hdf5(pickle_path, hdf5_path, optimizer_override="CMA_MO")

    with h5py.File(hdf5_path, "r") as hdf5_file:
        assert hdf5_file.attrs["optimizer"] == "CMA_MO"


class _FakeCell:
    name = "cell_a"
    recordings = [1, 2]


class _FakeProtocol:
    name = "IDrest"
    amplitude = 0.15
    recordings = [1]


def test_cells_pickle_hdf5_roundtrip(tmp_path):
    pickle_path = tmp_path / "cells.pkl"
    hdf5_path = tmp_path / "cells.h5"
    restored_pickle_path = tmp_path / "cells_restored.pkl"
    cells = [_FakeCell()]

    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(cells, pickle_file)

    pickle_to_hdf5(pickle_path, hdf5_path)
    hdf5_to_pickle(hdf5_path, restored_pickle_path)

    with open(restored_pickle_path, "rb") as pickle_file:
        restored = pickle.load(pickle_file, encoding="latin1")

    assert restored[0].name == "cell_a"
    assert len(restored[0].recordings) == 2


def test_protocols_pickle_hdf5_roundtrip(tmp_path):
    pickle_path = tmp_path / "protocols.pkl"
    hdf5_path = tmp_path / "protocols.h5"
    restored_pickle_path = tmp_path / "protocols_restored.pkl"
    protocols = [_FakeProtocol()]

    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(protocols, pickle_file)

    pickle_to_hdf5(pickle_path, hdf5_path)
    hdf5_to_pickle(hdf5_path, restored_pickle_path)

    with open(restored_pickle_path, "rb") as pickle_file:
        restored = pickle.load(pickle_file, encoding="latin1")

    assert restored[0].name == "IDrest"
    assert restored[0].amplitude == 0.15


def test_fitness_wrapper_properties():
    fitness = _Fitness([1.0, 2.0], weights=[-1.0, -1.0])

    assert fitness.valid is True
    assert fitness.reduce == 3.0
    assert fitness.weighted_reduce == -3.0


def test_load_checkpoint_hdf5_rejects_wrong_content_type(tmp_path):
    hdf5_path = tmp_path / "wrong.h5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.attrs["content_type"] = CONTENT_TYPE_CELLS
        hdf5_file.attrs["format_version"] = FORMAT_VERSION

    with pytest.raises(ValueError, match="does not contain an optimisation summary"):
        load_checkpoint_hdf5(hdf5_path)


def test_hdf5_to_pickle_rejects_unsupported_content_type(tmp_path):
    hdf5_path = tmp_path / "unsupported.h5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.attrs["content_type"] = "unknown"
        hdf5_file.attrs["format_version"] = FORMAT_VERSION

    with pytest.raises(ValueError, match="Unsupported content type"):
        hdf5_to_pickle(hdf5_path, tmp_path / "out.pkl")


def test_pickle_to_hdf5_with_explicit_content_type(tmp_path):
    pickle_path = tmp_path / "cells.pkl"
    hdf5_path = tmp_path / "cells.h5"
    cells = [_FakeCell()]
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(cells, pickle_file)

    pickle_to_hdf5(pickle_path, hdf5_path, content_type=CONTENT_TYPE_CELLS)

    with h5py.File(hdf5_path, "r") as hdf5_file:
        assert hdf5_file.attrs["content_type"] == CONTENT_TYPE_CELLS


def test_checkpoint_pickle_to_hdf5_without_history(tmp_path):
    pickle_path = tmp_path / "emodel=L5PC__seed=4.pkl"
    hdf5_path = tmp_path / "checkpoint.h5"
    checkpoint = {
        "generation": 0,
        "param_names": ["a"],
        "halloffame": [_Individual([1.0], fitness_values=[0.5])],
        "population": [],
    }
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(checkpoint, pickle_file)

    checkpoint_pickle_to_hdf5(pickle_path, hdf5_path)
    loaded = load_checkpoint_hdf5(hdf5_path)

    assert loaded["history"].genealogy_history == {}
