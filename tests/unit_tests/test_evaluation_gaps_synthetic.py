"""Deterministic tests for remaining evaluation gaps."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bluepyemodel.evaluation import evaluation as evaluation_module
from bluepyemodel.evaluation import fitness_calculator_configuration as fcc_module
from bluepyemodel.evaluation.fitness_calculator_configuration import (
    FitnessCalculatorConfiguration,
)
from bluepyemodel.evaluation.protocols import ProtocolRunner


def make_section(subtree_size):
    return SimpleNamespace(subtree=lambda: list(range(subtree_size)))


def make_cell(sections):
    return SimpleNamespace(icell=SimpleNamespace(apical=sections))


def test_set_morphology_dependent_locations_somadistanceapic(monkeypatch):
    monkeypatch.setattr(fcc_module, "seclist_to_sec", {"apical": "apic"}, raising=False)
    recording = {
        "type": "somadistanceapic",
        "seclist_name": "apical",
        "name": "prot.apic.v",
    }

    result = fcc_module._set_morphology_dependent_locations(recording, make_cell([]))

    assert len(result) == 1
    assert result[0]["sec_name"] == "apic"


def test_set_morphology_dependent_locations_terminal_and_all_sections():
    cell = make_cell([make_section(1), make_section(3), make_section(1)])
    terminal = {
        "type": "terminal_sections",
        "seclist_name": "apical",
        "name": "prot.apic.v",
    }
    all_sections = {
        "type": "all_sections",
        "seclist_name": "apical",
        "name": "prot.apic.v",
    }

    terminal_recs = fcc_module._set_morphology_dependent_locations(terminal, cell)
    all_recs = fcc_module._set_morphology_dependent_locations(all_sections, cell)

    assert [rec["sec_index"] for rec in terminal_recs] == [0, 2]
    assert [rec["name"] for rec in terminal_recs] == ["prot.apic_0.v", "prot.apic_2.v"]
    assert [rec["sec_index"] for rec in all_recs] == [0, 1, 2]
    assert all(rec["type"] == "nrnseclistcomp" for rec in all_recs + terminal_recs)


def test_set_morphology_dependent_locations_default_and_empty():
    default = {"type": "CompRecording", "seclist_name": "soma", "name": "prot.soma.v"}
    assert fcc_module._set_morphology_dependent_locations(default, make_cell([])) == [
        default
    ]

    empty = {"type": "all_sections", "seclist_name": "apical", "name": "prot.apic.v"}
    assert fcc_module._set_morphology_dependent_locations(empty, make_cell([])) == []


def make_configuration():
    configuration = FitnessCalculatorConfiguration.__new__(
        FitnessCalculatorConfiguration
    )
    configuration.protocols = []
    configuration.efeatures = []
    configuration.stochasticity = False
    configuration.workflow_id = "workflow-1"
    return configuration


def test_check_stochasticity_list_and_flag():
    configuration = make_configuration()
    configuration.stochasticity = ["IDrest_100"]

    assert configuration.check_stochasticity("IDrest_100") is True
    assert configuration.check_stochasticity("IV_40") is False

    configuration.stochasticity = True
    assert configuration.check_stochasticity("IV_40") is True


def test_remove_featureless_protocols_keeps_matching_only():
    configuration = make_configuration()
    configuration.protocols = [
        SimpleNamespace(name="IDrest_100"),
        SimpleNamespace(name="IV_40"),
    ]
    configuration.efeatures = [SimpleNamespace(protocol_name="IDrest_100")]

    configuration.remove_featureless_protocols()

    assert [p.name for p in configuration.protocols] == ["IDrest_100"]


def test_protocol_exist_and_related_nexus_ids():
    configuration = make_configuration()
    configuration.protocols = [SimpleNamespace(name="IDrest_100")]

    assert configuration.protocol_exist("IDrest_100") is True
    ids = configuration.get_related_nexus_ids()
    assert ids["generation"]["activity"]["followedWorkflow"]["id"] == "workflow-1"


def test_as_dict_and_str_representation():
    configuration = make_configuration()
    configuration.protocols = [SimpleNamespace(as_dict=lambda: {"name": "IDrest_100"})]
    configuration.efeatures = [SimpleNamespace(as_dict=lambda: {"name": "Spikecount"})]

    assert configuration.as_dict() == {
        "efeatures": [{"name": "Spikecount"}],
        "protocols": [{"name": "IDrest_100"}],
    }
    text = str(configuration)
    assert "Fitness Calculator Configuration" in text
    assert "IDrest_100" in text
    assert "Spikecount" in text


def test_initialise_protocols_and_efeatures_handle_none():
    configuration = make_configuration()
    configuration.ion_variables = None

    assert configuration.initialise_protocols(None) == []
    assert configuration.initialise_efeatures(None) == []


def test_init_from_legacy_dict_requires_rmp_protocol():
    configuration = FitnessCalculatorConfiguration(name_rmp_protocol="IDrest_100")

    with pytest.raises(ValueError, match="requested for RMP"):
        configuration.init_from_legacy_dict({"IV_40": {}}, {}, None)


def test_init_from_legacy_dict_requires_rin_protocol():
    configuration = FitnessCalculatorConfiguration(name_rin_protocol="IDrest_100")

    with pytest.raises(ValueError, match="requested for Rin"):
        configuration.init_from_legacy_dict({"IV_40": {}}, {}, None)


def test_get_responses_applies_threshold_data_and_unfreezes():
    protocol_runner = ProtocolRunner.__new__(ProtocolRunner)
    protocol_runner.threshold_data = {}
    other_protocol = SimpleNamespace()
    unfrozen = []
    evaluator = SimpleNamespace(
        cell_model=SimpleNamespace(unfreeze=unfrozen.append),
        fitness_protocols={"main": protocol_runner, "other": other_protocol},
        run_protocols=lambda protocols, param_values: {"Step.soma.v": [1.0]},
    )

    responses = evaluation_module.get_responses(
        {
            "evaluator": evaluator,
            "parameters": {"gNa": 0.1},
            "threshold_data": {"bpo_holding_current": 0.2},
        }
    )

    assert responses["evaluator"] is evaluator
    assert responses["Step.soma.v"] == [1.0]
    assert unfrozen == [{"gNa": 0.1}]
    assert protocol_runner.threshold_data == {"bpo_holding_current": 0.2}


def make_access_point(emodels, iteration=None, emodel="L5PC"):
    return SimpleNamespace(
        get_emodels=lambda: emodels,
        emodel_metadata=SimpleNamespace(emodel=emodel, iteration=iteration),
    )


def make_model(seed=1, iteration=None, passed_validation=None):
    return SimpleNamespace(
        seed=seed,
        parameters={"gNa": 0.1},
        threshold_data={"bpo_holding_current": 0.1},
        emodel_metadata=SimpleNamespace(
            iteration=iteration, as_string=lambda s: f"m__{s}"
        ),
        passed_validation=passed_validation,
        responses=None,
        evaluator=None,
    )


def test_compute_responses_filters_and_assigns(monkeypatch):
    model_a = make_model(seed=1)
    model_b = make_model(seed=2)
    access_point = make_access_point([model_a, model_b])
    evaluator = SimpleNamespace(label="evaluator")

    def fake_map(function, items):
        assert len(items) == 1
        return [{"Step.soma.v": [1.0], "evaluator": evaluator}]

    result = evaluation_module.compute_responses(
        access_point, evaluator, fake_map, seeds=[1]
    )

    assert result == [model_a]
    assert model_a.evaluator is evaluator
    assert model_a.responses == {"Step.soma.v": [1.0]}


def test_compute_responses_filters_iteration_and_validation(monkeypatch):
    matching = make_model(seed=1, iteration="iter1", passed_validation=None)
    other_iteration = make_model(seed=2, iteration="iter2")
    validated = make_model(seed=3, iteration="iter1", passed_validation=True)
    access_point = make_access_point(
        [matching, other_iteration, validated], iteration="iter1"
    )

    result = evaluation_module.compute_responses(
        access_point,
        SimpleNamespace(),
        lambda function, items: [{"evaluator": SimpleNamespace()} for _ in items],
        preselect_for_validation=True,
    )

    assert result == [matching]


def test_compute_responses_returns_empty_and_warns():
    access_point = make_access_point([])

    assert (
        evaluation_module.compute_responses(access_point, SimpleNamespace(), map) == []
    )


def test_compute_responses_stores_responses(monkeypatch):
    model_a = make_model()
    access_point = make_access_point([model_a])
    stored = []
    monkeypatch.setattr(evaluation_module, "locally_store_responses", stored.append)

    evaluation_module.compute_responses(
        access_point,
        SimpleNamespace(),
        lambda function, items: [{"evaluator": SimpleNamespace()}],
        store_responses=True,
    )

    assert stored == [model_a]


def test_compute_responses_loads_from_local(monkeypatch):
    model_a = make_model()
    access_point = make_access_point([model_a])
    monkeypatch.setattr(
        evaluation_module, "check_local_responses_presence", lambda *a: True
    )
    monkeypatch.setattr(
        evaluation_module,
        "load_responses_from_local_files",
        lambda emodels, cell_eval: [{"loaded": True, "evaluator": SimpleNamespace()}],
    )
    should_not_run = MagicMock()

    evaluation_module.compute_responses(
        access_point, SimpleNamespace(), should_not_run, load_from_local=True
    )

    should_not_run.assert_not_called()
    assert model_a.responses == {"loaded": True}


def test_compute_responses_recomputes_threshold_protocols():
    model_a = make_model()
    access_point = make_access_point([model_a])
    captured = {}

    def fake_map(function, items):
        captured["threshold_data"] = items[0]["threshold_data"]
        return [{"evaluator": SimpleNamespace()}]

    evaluation_module.compute_responses(
        access_point, SimpleNamespace(), fake_map, recompute_threshold_protocols=True
    )

    assert captured["threshold_data"] == {}
