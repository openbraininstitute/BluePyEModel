"""Convert BluePyEModel pickle files to and from HDF5."""

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
import pickle
import re
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2

CONTENT_TYPE_OPTIMISATION_SUMMARY = "optimisation_summary"
CONTENT_TYPE_CELLS = "cells"
CONTENT_TYPE_PROTOCOLS = "protocols"

CHECKPOINT_KEYS = (
    "generation",
    "param_names",
    "halloffame",
    "population",
    "logbook",
    "history",
    "CMA_es",
)


class _Fitness:
    """Stand-in for deap.base.Fitness / WeightedReducedFitness."""

    def __init__(self, values, weights=None):
        self.values = tuple(values)
        self.weights = tuple(weights) if weights is not None else tuple(-1.0 for _ in values)
        self.wvalues = tuple(w * v for w, v in zip(self.weights, self.values))

    @property
    def valid(self):
        return len(self.wvalues) != 0

    @property
    def reduce(self):
        return sum(self.values)

    @property
    def weighted_reduce(self):
        return sum(self.wvalues)


class _Individual(list):
    """Stand-in for a DEAP Individual (list of genes with .fitness attribute)."""

    def __init__(self, genes, fitness_values=None, fitness_weights=None):
        super().__init__(genes)
        if fitness_values is not None:
            self.fitness = _Fitness(fitness_values, fitness_weights)


class _GeneOnlyIndividual(list):
    """Lightweight individual for genealogy_history — only genes, no fitness."""

    def __init__(self, genes):
        super().__init__(genes)


class _Logbook:
    """Stand-in for deap.tools.Logbook supporting .select() and .header."""

    def __init__(self, data, header):
        self._data = data
        self.header = header

    def select(self, field):
        return list(self._data.get(field, []))


class _CMAStatus:
    """Stand-in for CMA_SO / CMA_MO that exposes .active and .check_termination()."""

    def __init__(self, active):
        self.active = active

    def check_termination(self, gen):
        """No-op. The active flag was determined at checkpoint save time."""


class _History:
    """Stand-in for deap.tools.History with .genealogy_history dict."""

    def __init__(self, genealogy_history=None):
        self.genealogy_history = genealogy_history if genealogy_history is not None else {}


def _store_pickled(group, name, obj):
    group.create_dataset(name, data=np.void(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))


def _load_pickled(dataset):
    return pickle.loads(bytes(dataset[()]))


def _load_pickle(pickle_path):
    with open(pickle_path, "rb") as pickle_file:
        return pickle.load(pickle_file, encoding="latin1")


def _dump_pickle(pickle_path, data):
    pickle_path = Path(pickle_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def _validate_format_version(hdf5_file):
    if hdf5_file.attrs.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported HDF5 conversion format version.")


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _store_string_dataset(group, name, values):
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_string_dtype())


def _load_string_dataset(dataset):
    values = []
    for value in dataset[:]:
        if isinstance(value, (bytes, bytearray)):
            values.append(value.decode("utf-8"))
        else:
            values.append(str(value))
    return values


def _infer_content_type(data, pickle_path):
    if isinstance(data, dict) and {"generation", "population", "halloffame"}.issubset(data):
        return CONTENT_TYPE_OPTIMISATION_SUMMARY

    if isinstance(data, list) and data:
        type_name = type(data[0]).__name__
        if type_name == "Cell":
            return CONTENT_TYPE_CELLS
        if type_name == "Protocol":
            return CONTENT_TYPE_PROTOCOLS

    filename = Path(pickle_path).name
    if filename == "cells.pkl":
        return CONTENT_TYPE_CELLS
    if filename == "protocols.pkl":
        return CONTENT_TYPE_PROTOCOLS

    raise ValueError(
        "Could not infer pickle content type. Pass content_type explicitly or use a "
        "supported filename (cells.pkl, protocols.pkl)."
    )


def _get_content_type(hdf5_file):
    if "content_type" not in hdf5_file.attrs:
        raise ValueError("HDF5 file is missing required attribute 'content_type'.")
    return hdf5_file.attrs["content_type"]


def _detect_optimizer(checkpoint):
    if "CMA_es" in checkpoint:
        class_name = type(checkpoint["CMA_es"]).__name__
        if "MO" in class_name or "MultiObjective" in class_name:
            return "CMA_MO"
        return "CMA_SO"
    if "parents" in checkpoint:
        return "IBEA"
    return "IBEA"


def _extract_seed_from_path(path):
    match = re.search(r"seed=(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _individuals_to_arrays(individuals):
    if not individuals:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))
    genes = np.asarray([list(individual) for individual in individuals], dtype=np.float64)
    fitness_values = np.asarray(
        [list(individual.fitness.values) for individual in individuals],
        dtype=np.float64,
    )
    fitness_reduce = np.asarray(
        [sum(individual.fitness.values) for individual in individuals],
        dtype=np.float64,
    )
    return genes, fitness_values, fitness_reduce


def _store_individuals_group(group, individuals):
    genes, fitness_values, fitness_reduce = _individuals_to_arrays(individuals)
    group.attrs["size"] = len(individuals)
    group.create_dataset("genes", data=genes)
    group.create_dataset("fitness_values", data=fitness_values)
    group.create_dataset("fitness_reduce", data=fitness_reduce)


def _load_individuals_group(group, fitness_weights=None):
    genes = group["genes"][:]
    if genes.shape[0] == 0:
        return []

    fitness_values = group["fitness_values"][:]
    individuals = []
    for row_index in range(genes.shape[0]):
        individuals.append(
            _Individual(
                genes=genes[row_index].tolist(),
                fitness_values=fitness_values[row_index].tolist(),
                fitness_weights=fitness_weights,
            )
        )
    return individuals


def _store_logbook(group, logbook):
    header = logbook.header if hasattr(logbook, "header") else []
    group.attrs["header"] = header
    for field in header:
        data = logbook.select(field)
        if not data:
            continue
        if field in ("gen", "nevals"):
            group.create_dataset(field, data=np.asarray(data, dtype=np.int64))
        else:
            group.create_dataset(field, data=np.asarray(data, dtype=np.float64))


def _load_logbook(group):
    header = list(group.attrs.get("header", []))
    data = {}
    for field in header:
        if field in group:
            data[field] = group[field][:].tolist()
    return _Logbook(data, header)


def _store_history(group, history):
    genealogy_history = history.genealogy_history
    if not genealogy_history:
        group.create_dataset("genealogy_genes", data=np.empty((0, 0), dtype=np.float64))
        return

    max_id = max(genealogy_history)
    n_params = len(genealogy_history[1])
    genes = np.empty((max_id, n_params), dtype=np.float64)
    for individual_id in range(1, max_id + 1):
        genes[individual_id - 1] = list(genealogy_history[individual_id])
    group.create_dataset("genealogy_genes", data=genes)


def _load_history(group):
    genes = group["genealogy_genes"][:]
    if genes.size == 0:
        return _History({})

    genealogy = {
        individual_id + 1: _GeneOnlyIndividual(genes[individual_id].tolist())
        for individual_id in range(genes.shape[0])
    }
    return _History(genealogy)


def checkpoint_pickle_to_hdf5(pickle_path, hdf5_path, optimizer_override=None):
    """Convert an optimisation checkpoint pickle file to HDF5."""
    pickle_path = Path(pickle_path)
    hdf5_path = Path(hdf5_path)
    checkpoint = _load_pickle(pickle_path)

    optimizer = optimizer_override or _detect_optimizer(checkpoint)
    generation = checkpoint["generation"]
    param_names = checkpoint.get("param_names", [])
    halloffame = list(checkpoint["halloffame"]) if checkpoint.get("halloffame") else []
    population = checkpoint.get("population", [])
    logbook = checkpoint.get("logbook")
    history = checkpoint.get("history")
    seed = _extract_seed_from_path(pickle_path)

    cma_active = True
    if "CMA_es" in checkpoint:
        cma_active = checkpoint["CMA_es"].active

    n_params = len(param_names) if param_names else (len(population[0]) if population else 0)
    sample_individual = halloffame[0] if halloffame else (population[0] if population else None)
    n_objectives = len(sample_individual.fitness.values) if sample_individual else 0
    fitness_weights = (
        np.asarray(sample_individual.fitness.weights, dtype=np.float64)
        if sample_individual
        else np.empty(0)
    )

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.attrs["content_type"] = CONTENT_TYPE_OPTIMISATION_SUMMARY
        hdf5_file.attrs["format_version"] = FORMAT_VERSION
        hdf5_file.attrs["optimizer"] = optimizer
        hdf5_file.attrs["generation"] = generation
        hdf5_file.attrs["seed"] = seed
        hdf5_file.attrs["n_params"] = n_params
        hdf5_file.attrs["n_objectives"] = n_objectives
        hdf5_file.attrs["fitness_weights"] = fitness_weights
        hdf5_file.attrs["cma_active"] = cma_active

        _store_string_dataset(hdf5_file, "param_names", param_names)
        _store_individuals_group(hdf5_file.create_group("halloffame"), halloffame)
        _store_individuals_group(hdf5_file.create_group("population"), population)

        if logbook is not None:
            _store_logbook(hdf5_file.create_group("logbook"), logbook)

        history_group = hdf5_file.create_group("history")
        if history is not None:
            _store_history(history_group, history)
        else:
            history_group.create_dataset("genealogy_genes", data=np.empty((0, 0), dtype=np.float64))

    logger.info(
        "Written: %s (optimizer=%s, generation=%d, seed=%d, n_params=%d, "
        "n_objectives=%d, halloffame=%d, population=%d)",
        hdf5_path,
        optimizer,
        generation,
        seed,
        n_params,
        n_objectives,
        len(halloffame),
        len(population),
    )


def load_checkpoint_hdf5(hdf5_path):
    """Load an HDF5 optimisation summary and return a checkpoint-compatible dict."""
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as hdf5_file:
        _validate_format_version(hdf5_file)
        if _get_content_type(hdf5_file) != CONTENT_TYPE_OPTIMISATION_SUMMARY:
            raise ValueError("HDF5 file does not contain an optimisation summary.")

        optimizer = str(hdf5_file.attrs["optimizer"])
        generation = int(hdf5_file.attrs["generation"])
        cma_active = bool(hdf5_file.attrs.get("cma_active", True))

        fitness_weights = None
        if "fitness_weights" in hdf5_file.attrs:
            weights = hdf5_file.attrs["fitness_weights"]
            if hasattr(weights, "__len__") and len(weights) > 0:
                fitness_weights = weights.tolist()

        checkpoint = {
            "generation": generation,
            "param_names": _load_string_dataset(hdf5_file["param_names"]),
            "halloffame": _load_individuals_group(hdf5_file["halloffame"], fitness_weights),
            "population": _load_individuals_group(hdf5_file["population"], fitness_weights),
            "logbook": _load_logbook(hdf5_file["logbook"]) if "logbook" in hdf5_file else None,
            "history": (
                _load_history(hdf5_file["history"]) if "history" in hdf5_file else _History({})
            ),
        }

        if "CMA" in optimizer:
            checkpoint["CMA_es"] = _CMAStatus(active=cma_active)

    return checkpoint


def checkpoint_hdf5_to_pickle(hdf5_path, pickle_path):
    """Convert an optimisation summary HDF5 file back to a pickle checkpoint dict."""
    checkpoint = load_checkpoint_hdf5(hdf5_path)
    checkpoint = {key: checkpoint[key] for key in CHECKPOINT_KEYS if key in checkpoint}
    _dump_pickle(pickle_path, checkpoint)


def cells_pickle_to_hdf5(pickle_path, hdf5_path):
    """Convert a BluePyEfe cells pickle file to HDF5."""
    pickle_path = Path(pickle_path)
    hdf5_path = Path(hdf5_path)
    cells = _load_pickle(pickle_path)

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.attrs["format_version"] = FORMAT_VERSION
        hdf5_file.attrs["content_type"] = CONTENT_TYPE_CELLS
        hdf5_file.attrs["n_items"] = len(cells)

        _store_string_dataset(hdf5_file, "names", [cell.name for cell in cells])
        hdf5_file.create_dataset("recordings_count", data=[len(cell.recordings) for cell in cells])
        _store_pickled(hdf5_file, "data", cells)


def cells_hdf5_to_pickle(hdf5_path, pickle_path):
    """Convert a BluePyEfe cells HDF5 file back to pickle."""
    hdf5_path = Path(hdf5_path)
    pickle_path = Path(pickle_path)

    with h5py.File(hdf5_path, "r") as hdf5_file:
        _validate_format_version(hdf5_file)
        if _get_content_type(hdf5_file) != CONTENT_TYPE_CELLS:
            raise ValueError("HDF5 file does not contain cells data.")
        cells = _load_pickled(hdf5_file["data"])

    _dump_pickle(pickle_path, cells)


def protocols_pickle_to_hdf5(pickle_path, hdf5_path):
    """Convert a BluePyEfe protocols pickle file to HDF5."""
    pickle_path = Path(pickle_path)
    hdf5_path = Path(hdf5_path)
    protocols = _load_pickle(pickle_path)

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.attrs["format_version"] = FORMAT_VERSION
        hdf5_file.attrs["content_type"] = CONTENT_TYPE_PROTOCOLS
        hdf5_file.attrs["n_items"] = len(protocols)

        _store_string_dataset(hdf5_file, "names", [protocol.name for protocol in protocols])
        hdf5_file.create_dataset("amplitudes", data=[protocol.amplitude for protocol in protocols])
        hdf5_file.create_dataset(
            "recordings_count",
            data=[len(protocol.recordings) for protocol in protocols],
        )
        _store_pickled(hdf5_file, "data", protocols)


def protocols_hdf5_to_pickle(hdf5_path, pickle_path):
    """Convert a BluePyEfe protocols HDF5 file back to pickle."""
    hdf5_path = Path(hdf5_path)
    pickle_path = Path(pickle_path)

    with h5py.File(hdf5_path, "r") as hdf5_file:
        _validate_format_version(hdf5_file)
        if _get_content_type(hdf5_file) != CONTENT_TYPE_PROTOCOLS:
            raise ValueError("HDF5 file does not contain protocols data.")
        protocols = _load_pickled(hdf5_file["data"])

    _dump_pickle(pickle_path, protocols)


_CONTENT_CONVERTERS = {
    CONTENT_TYPE_OPTIMISATION_SUMMARY: (checkpoint_pickle_to_hdf5, checkpoint_hdf5_to_pickle),
    CONTENT_TYPE_CELLS: (cells_pickle_to_hdf5, cells_hdf5_to_pickle),
    CONTENT_TYPE_PROTOCOLS: (protocols_pickle_to_hdf5, protocols_hdf5_to_pickle),
}


def pickle_to_hdf5(pickle_path, hdf5_path, content_type=None, optimizer_override=None):
    """Convert a supported pickle file to HDF5, inferring content type when omitted."""
    pickle_path = Path(pickle_path)
    if content_type is None:
        content_type = _infer_content_type(_load_pickle(pickle_path), pickle_path)
    if content_type not in _CONTENT_CONVERTERS:
        raise ValueError(f"Unsupported content type: {content_type}")
    converter = _CONTENT_CONVERTERS[content_type][0]
    if content_type == CONTENT_TYPE_OPTIMISATION_SUMMARY:
        converter(pickle_path, hdf5_path, optimizer_override=optimizer_override)
    else:
        converter(pickle_path, hdf5_path)


def hdf5_to_pickle(hdf5_path, pickle_path):
    """Convert a supported HDF5 file back to pickle."""
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as hdf5_file:
        _validate_format_version(hdf5_file)
        content_type = _get_content_type(hdf5_file)
    if content_type not in _CONTENT_CONVERTERS:
        raise ValueError(f"Unsupported content type: {content_type}")
    _CONTENT_CONVERTERS[content_type][1](hdf5_path, pickle_path)
