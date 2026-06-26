"""HDF5 checkpoint reader for BluePyOpt optimisation results.

Provides read_checkpoint_h5() which returns a dict compatible with all
BluePyEModel analysis functions (store_best_model, optimisation plot,
evolution_parameters_density, check_optimisation_state).

Also provides convert_checkpoint() for pickle → HDF5 conversion.
"""

import logging
import pickle
import re
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2


# =============================================================================
# Shim classes — lightweight stand-ins for DEAP objects so that BluePyEModel's
# analysis pipeline can consume HDF5 data without modification.
# =============================================================================


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
    """Lightweight individual for genealogy_history — only genes, no fitness.

    Used by evolution_parameters_density which only does ind[param_index].
    """

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
        pass


class _History:
    """Stand-in for deap.tools.History with .genealogy_history dict."""

    def __init__(self, genealogy_history=None):
        self.genealogy_history = genealogy_history if genealogy_history is not None else {}


# =============================================================================
# HDF5 → checkpoint dict reader
# =============================================================================


def _read_individuals_from_group(group, fitness_weights=None):
    """Reconstruct _Individual objects from an HDF5 group."""
    genes = group["genes"][:]
    fitness_values = group["fitness_values"][:]
    individuals = []
    for i in range(genes.shape[0]):
        ind = _Individual(
            genes=genes[i].tolist(),
            fitness_values=fitness_values[i].tolist(),
            fitness_weights=fitness_weights,
        )
        individuals.append(ind)
    return individuals


def _read_logbook_from_group(group):
    """Reconstruct a _Logbook from an HDF5 group."""
    header = list(group.attrs.get("header", []))
    data = {}
    for field in header:
        if field in group:
            data[field] = group[field][:].tolist()
    return _Logbook(data, header)


def _read_history_from_group(group):
    """Reconstruct a _History from an HDF5 group."""
    genes = group["genealogy_genes"][:]
    if genes.size == 0:
        return _History({})

    genealogy = {}
    for i in range(genes.shape[0]):
        genealogy[i + 1] = _GeneOnlyIndividual(genes[i].tolist())
    return _History(genealogy)


def read_checkpoint_h5(h5_path):
    """Read an HDF5 checkpoint and return a dict mimicking a pickle checkpoint.

    Compatible with all BluePyEModel analysis consumers:
      - store_best_model
      - optimisation() plot
      - evolution_parameters_density()
      - access_point.__str__
      - check_optimisation_state

    Returns:
        (run_dict, seed)
    """
    with h5py.File(h5_path, "r") as h5:
        optimizer = str(h5.attrs["optimizer"])
        generation = int(h5.attrs["generation"])
        seed = int(h5.attrs["seed"])
        cma_active = bool(h5.attrs.get("cma_active", True))

        fitness_weights = None
        if "fitness_weights" in h5.attrs:
            fw = h5.attrs["fitness_weights"]
            if hasattr(fw, "__len__") and len(fw) > 0:
                fitness_weights = fw.tolist()

        param_names = [
            s.decode("utf-8") if isinstance(s, bytes) else s
            for s in h5["param_names"][:]
        ]

        halloffame = _read_individuals_from_group(h5["halloffame"], fitness_weights)
        population = _read_individuals_from_group(h5["population"], fitness_weights)

        logbook = None
        if "logbook" in h5:
            logbook = _read_logbook_from_group(h5["logbook"])

        history = _History({})
        if "history" in h5:
            history = _read_history_from_group(h5["history"])

    run = {
        "generation": generation,
        "param_names": param_names,
        "halloffame": halloffame,
        "population": population,
        "logbook": logbook,
        "history": history,
    }

    if "CMA" in optimizer:
        run["CMA_es"] = _CMAStatus(active=cma_active)

    return run, seed


# =============================================================================
# Pickle → HDF5 conversion
# =============================================================================


def detect_optimizer(cp):
    """Auto-detect the optimizer type from checkpoint keys."""
    if "CMA_es" in cp:
        class_name = type(cp["CMA_es"]).__name__
        if "MO" in class_name or "MultiObjective" in class_name:
            return "CMA_MO"
        return "CMA_SO"
    if "parents" in cp:
        return "IBEA"
    return "IBEA"


def _extract_seed_from_path(path):
    """Extract seed number from checkpoint filename (pattern: seed=<N>)."""
    match = re.search(r"seed=(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _individuals_to_arrays(individuals):
    """Convert DEAP individuals to numpy arrays (genes + fitness)."""
    if not individuals:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))
    genes = np.array([list(ind) for ind in individuals], dtype=np.float64)
    fitness_values = np.array(
        [list(ind.fitness.values) for ind in individuals], dtype=np.float64
    )
    fitness_reduce = np.array(
        [sum(ind.fitness.values) for ind in individuals], dtype=np.float64
    )
    return genes, fitness_values, fitness_reduce


def _write_individuals_group(group, individuals):
    """Write individuals (genes + fitness) to an HDF5 group."""
    genes, fitness_values, fitness_reduce = _individuals_to_arrays(individuals)
    group.attrs["size"] = len(individuals)
    group.create_dataset("genes", data=genes)
    group.create_dataset("fitness_values", data=fitness_values)
    group.create_dataset("fitness_reduce", data=fitness_reduce)


def _write_logbook(group, logbook):
    """Write logbook statistics to an HDF5 group."""
    header = logbook.header if hasattr(logbook, "header") else []
    group.attrs["header"] = header
    for field in header:
        data = logbook.select(field)
        if not data:
            continue
        if field in ("gen", "nevals"):
            group.create_dataset(field, data=np.array(data, dtype=np.int64))
        else:
            group.create_dataset(field, data=np.array(data, dtype=np.float64))


def _write_history(group, history):
    """Write genealogy_history genes to an HDF5 group."""
    gh = history.genealogy_history
    if not gh:
        group.create_dataset("genealogy_genes", data=np.empty((0, 0), dtype=np.float64))
        return

    n = max(gh.keys())
    n_params = len(gh[1])
    genes = np.empty((n, n_params), dtype=np.float64)
    for idx in range(1, n + 1):
        genes[idx - 1] = list(gh[idx])

    group.create_dataset("genealogy_genes", data=genes)


def convert_checkpoint(pickle_path, output_path=None, optimizer_override=None):
    """Convert a BluePyOpt pickle checkpoint to an HDF5 file.

    Args:
        pickle_path (str): Path to the .pkl checkpoint file.
        output_path (str): Path for the output .h5 file (default: same name with .h5).
        optimizer_override (str): Force optimizer type ("IBEA", "CMA_SO", "CMA_MO").

    Returns:
        str: Path to the created HDF5 file.
    """
    pickle_path = Path(pickle_path)
    if output_path is None:
        output_path = pickle_path.with_suffix(".h5")
    else:
        output_path = Path(output_path)

    with open(pickle_path, "rb") as f:
        cp = pickle.load(f, encoding="latin1")

    optimizer = optimizer_override or detect_optimizer(cp)
    logger.info("Detected optimizer: %s", optimizer)

    generation = cp["generation"]
    param_names = cp.get("param_names", [])
    halloffame = list(cp["halloffame"]) if cp.get("halloffame") else []
    population = cp.get("population", [])
    logbook = cp.get("logbook")
    history = cp.get("history")
    seed = _extract_seed_from_path(str(pickle_path))

    cma_active = True
    if "CMA_es" in cp:
        cma_active = cp["CMA_es"].active

    n_params = len(param_names) if param_names else (len(population[0]) if population else 0)
    sample_ind = halloffame[0] if halloffame else (population[0] if population else None)
    n_objectives = len(sample_ind.fitness.values) if sample_ind else 0
    fitness_weights = np.array(
        sample_ind.fitness.weights, dtype=np.float64
    ) if sample_ind else np.empty(0)

    with h5py.File(output_path, "w") as h5:
        h5.attrs["content_type"] = "optimisation_summary"
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["optimizer"] = optimizer
        h5.attrs["generation"] = generation
        h5.attrs["seed"] = seed
        h5.attrs["n_params"] = n_params
        h5.attrs["n_objectives"] = n_objectives
        h5.attrs["fitness_weights"] = fitness_weights
        h5.attrs["cma_active"] = cma_active

        dt = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("param_names", data=param_names, dtype=dt)

        hof_grp = h5.create_group("halloffame")
        _write_individuals_group(hof_grp, halloffame)

        pop_grp = h5.create_group("population")
        _write_individuals_group(pop_grp, population)

        if logbook is not None:
            lb_grp = h5.create_group("logbook")
            _write_logbook(lb_grp, logbook)

        hist_grp = h5.create_group("history")
        if history is not None:
            _write_history(hist_grp, history)
        else:
            hist_grp.create_dataset(
                "genealogy_genes", data=np.empty((0, 0), dtype=np.float64)
            )

    logger.info(
        "Written: %s (optimizer=%s, generation=%d, seed=%d, n_params=%d, "
        "n_objectives=%d, halloffame=%d, population=%d)",
        output_path, optimizer, generation, seed, n_params,
        n_objectives, len(halloffame), len(population),
    )
    return str(output_path)
