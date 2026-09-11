"""Deterministic tests for protocol orchestration and threshold searches."""

from types import SimpleNamespace

import pytest

from bluepyemodel.evaluation import protocols as protocols_module
from bluepyemodel.evaluation.protocols import (
    BPEMProtocol,
    ProtocolRunner,
    ResponseDependencies,
    RinProtocol,
    RMPProtocol,
    SearchHoldingCurrent,
    SearchThresholdCurrent,
    ThresholdBasedProtocol,
)


def test_response_dependencies_run_returns_none_responses():
    class Dependent(ResponseDependencies):
        def return_none_responses(self):
            return {"recording": None}

        def _run(
            self, cell_model, param_values=None, sim=None, isolate=None, timeout=None
        ):
            return {"recording": [1.0]}

    dependent = Dependent({"value": ["source", "source_key"]})

    assert dependent.run(None, responses={}) == {"recording": None}
    assert dependent.run(None, responses={"source_key": 1.0}) == {"recording": [1.0]}
    assert dependent.value == 1.0


def test_response_dependencies_base_methods_raise():
    dependency = ResponseDependencies()

    with pytest.raises(NotImplementedError):
        dependency.return_none_responses()
    with pytest.raises(NotImplementedError):
        dependency._run(None)


def test_threshold_based_protocol_returns_none_responses():
    protocol = ThresholdBasedProtocol.__new__(ThresholdBasedProtocol)
    protocol.recordings = [
        SimpleNamespace(name="Step.soma.v"),
        SimpleNamespace(name="Step.dend.v"),
    ]

    assert protocol.return_none_responses() == {
        "Step.soma.v": None,
        "Step.dend.v": None,
    }


def test_search_threshold_current_return_none_responses():
    search = SearchThresholdCurrent.__new__(SearchThresholdCurrent)
    search.output_key = "bpo_threshold_current"
    search.recording_name = "SearchThresholdCurrent.soma.v"

    assert search.return_none_responses() == {
        "bpo_threshold_current": None,
        "SearchThresholdCurrent.soma.v": None,
    }


def make_threshold_search(spikecounts, current_precision=1e-2, max_depth=10):
    search = SearchThresholdCurrent.__new__(SearchThresholdCurrent)
    search.output_key = "bpo_threshold_current"
    search.hold_key = "bpo_holding_current"
    search.recording_name = "SearchThresholdCurrent.soma.v"
    search.current_precision = current_precision
    search.max_depth = max_depth
    search.no_spikes = True
    search.max_threshold_voltage = -30.0
    search.rmp = -70.0
    search.rin = 100.0
    search._get_spikecount = lambda current, *args: spikecounts(current)
    return search


def test_define_search_bounds_rejects_silent_upper_bound():
    search = make_threshold_search(lambda current: 0)

    assert search.define_search_bounds(
        None, {}, None, None, {"bpo_holding_current": -0.1}
    ) == (
        None,
        None,
    )


def test_define_search_bounds_lowers_bound_when_spikes_allowed():
    search = make_threshold_search(lambda current: 1)
    search.no_spikes = False

    lower, upper = search.define_search_bounds(
        None, {}, None, None, {"bpo_holding_current": -0.1}
    )

    assert lower == pytest.approx(-0.6)
    assert upper == pytest.approx(0.4)


def test_define_search_bounds_handles_inverted_bounds():
    search = make_threshold_search(lambda current: 1)
    search.no_spikes = False

    lower, upper = search.define_search_bounds(
        None, {}, None, None, {"bpo_holding_current": 5.0}
    )

    assert lower == upper == pytest.approx(4.5)


def test_threshold_bisection_search_converges_on_spiking_side():
    search = make_threshold_search(
        lambda current: int(current > 0.25), current_precision=0.01
    )

    result = search.bisection_search(
        None, {}, None, None, upper_bound=1.0, lower_bound=0.0
    )

    assert result == pytest.approx(0.25, abs=0.02)


def test_threshold_bisection_search_stops_at_max_depth():
    search = make_threshold_search(
        lambda current: 0, current_precision=1e-9, max_depth=0
    )

    result = search.bisection_search(
        None, {}, None, None, upper_bound=1.0, lower_bound=0.0
    )

    assert result == pytest.approx(1.0)


def test_search_threshold_current_run_returns_none_without_dependencies():
    search = make_threshold_search(lambda current: 1)
    search.dependencies = {"rin": ["RinProtocol", "bpo_rin"]}

    assert search.run(None, responses={}) == {
        "bpo_threshold_current": None,
        "SearchThresholdCurrent.soma.v": None,
    }


def test_search_threshold_current_run_returns_none_for_bad_bounds():
    search = make_threshold_search(lambda current: 0)
    search.dependencies = {}

    assert search.run(None, responses={"bpo_holding_current": -0.1}) == {
        "bpo_threshold_current": None
    }


def test_search_threshold_current_run_appends_trace(monkeypatch):
    search = make_threshold_search(lambda current: int(current > 0.1))
    search.dependencies = {}
    search.stimulus = SimpleNamespace(amp=None)
    monkeypatch.setattr(search, "bisection_search", lambda *a, **k: 0.15)
    monkeypatch.setattr(
        search,
        "_run",
        lambda cell_model, param_values, sim=None, isolate=None, timeout=None: {
            "SearchThresholdCurrent.soma.v": [1.0]
        },
    )

    response = search.run(None, responses={"bpo_holding_current": -0.1})

    assert response["bpo_threshold_current"] == pytest.approx(0.15)
    assert response["SearchThresholdCurrent.soma.v"] == [1.0]
    assert search.stimulus.amp == pytest.approx(0.15)


def test_search_threshold_current_run_returns_none_threshold(monkeypatch):
    search = make_threshold_search(lambda current: int(current > 0.1))
    search.dependencies = {}
    monkeypatch.setattr(search, "bisection_search", lambda *a, **k: None)

    assert search.run(None, responses={"bpo_holding_current": -0.1}) == {
        "bpo_threshold_current": None
    }


def test_max_threshold_current_is_capped_at_two():
    search = SearchThresholdCurrent.__new__(SearchThresholdCurrent)
    search.max_threshold_voltage = -30.0
    search.rmp = -70.0
    search.rin = 100.0

    assert search.max_threshold_current() == pytest.approx(0.4)


def make_holding_search(voltages, precision=0.1, max_depth=7):
    search = SearchHoldingCurrent.__new__(SearchHoldingCurrent)
    search.output_key = "bpo_holding_current"
    search.recording_name = "SearchHoldingCurrent.soma.v"
    search.holding_voltage = -70.0
    search.voltage_precision = precision
    search.max_depth = max_depth
    search.upper_bound = 0.2
    search.lower_bound = -0.2
    search.strict_bounds = True
    search.target_voltage = SimpleNamespace(exp_mean=-70.0)
    search.get_voltage_base = lambda **kwargs: voltages(kwargs["holding_current"])
    return search


def test_holding_bisection_search_moves_to_lower_side_when_spiking():
    calls = []

    def voltages(current):
        calls.append(current)
        return None if current > -0.15 else -70.0

    search = make_holding_search(voltages, max_depth=6)

    result = search.bisection_search(
        None, {}, None, None, upper_bound=0.2, lower_bound=-0.2
    )

    assert result == pytest.approx(-0.15, abs=0.01)
    assert len(calls) > 1


def test_holding_search_run_returns_none_without_convergence(monkeypatch):
    search = make_holding_search(lambda current: -70.0)
    monkeypatch.setattr(search, "bisection_search", lambda *a, **k: None)

    assert search.run(None) == {"bpo_holding_current": None}


def test_holding_search_run_enlarges_non_strict_bounds(monkeypatch):
    search = make_holding_search(lambda current: -70.0)
    search.strict_bounds = False
    observed = {}

    def fake_bisection(cell_model, param_values, sim=None, isolate=None, **kwargs):
        observed.update(kwargs)
        return 0.0

    monkeypatch.setattr(search, "bisection_search", fake_bisection)
    monkeypatch.setattr(
        SearchHoldingCurrent,
        "run",
        SearchHoldingCurrent.run,
    )
    monkeypatch.setattr(
        "bluepyemodel.evaluation.protocols.BPEMProtocol.run",
        lambda self, cell_model, param_values, sim=None, isolate=None, timeout=None: {
            "SearchHoldingCurrent.soma.v": [1.0]
        },
    )

    response = search.run(None)

    assert response["bpo_holding_current"] == pytest.approx(0.0)
    assert response["SearchHoldingCurrent.soma.v"] == [1.0]
    assert observed["lower_bound"] == pytest.approx(-0.2)


class FakeProtocol:
    def __init__(self, name, result, dependencies=None, output_key=None):
        self.name = name
        self.result = result
        self.dependencies = dependencies or {}
        if output_key is not None:
            self.output_key = output_key
        self.calls = []

    def run(self, cell_model, param_values, sim, isolate, timeout, responses):
        self.calls.append(responses.copy())
        return self.result


def test_protocol_runner_str_lists_subprotocols():
    runner = ProtocolRunner({"a": FakeProtocol("a", {}), "b": FakeProtocol("b", {})})

    text = str(runner)

    assert "Sequence protocol ProtocolRunner" in text
    assert "2 subprotocols" in text
    assert "a" in text and "b" in text


def test_protocol_runner_transitive_dependency_order():
    first = FakeProtocol("first", {"first_key": 1})
    second = FakeProtocol(
        "second", {"second_key": 2}, dependencies={"v": ["first", "first_key"]}
    )
    third = FakeProtocol(
        "third", {"third_key": 3}, dependencies={"v": ["second", "second_key"]}
    )
    runner = ProtocolRunner({"third": third, "second": second, "first": first})

    assert runner.execution_order == ["first", "second", "third"]


def test_protocol_runner_saved_threshold_data_builds_set_instead_of_dict():
    """Documents a production defect in ProtocolRunner.run.

    When a protocol's ``output_key`` is already present in ``threshold_data``, the skip branch
    assigns ``new_responses = {key, value}`` (a set literal) instead of a dict. The subsequent
    ``responses.update(new_responses)`` then fails while unpacking the set members.
    """
    cached = FakeProtocol(
        "cached", {"bpo_holding_current": 5}, output_key="bpo_holding_current"
    )
    runner = ProtocolRunner({"cached": cached})
    runner.threshold_data = {"bpo_holding_current": -0.05}
    cell_model = SimpleNamespace(freeze=lambda values: None, unfreeze=lambda keys: None)

    with pytest.raises((TypeError, ValueError)):
        runner.run(cell_model, {})

    assert cached.calls == []


def test_protocol_runner_stops_when_protocol_returns_none():
    first = FakeProtocol("first", None)
    second = FakeProtocol("second", {"second_key": 1})
    runner = ProtocolRunner({"first": first, "second": second})
    cell_model = SimpleNamespace(freeze=lambda values: None, unfreeze=lambda keys: None)

    assert runner.run(cell_model, {}) == {}
    assert second.calls == []


def test_bpem_protocol_instantiate_uses_generic_fake_objects(monkeypatch):
    class Stimulus:
        def __init__(self):
            self.calls = []

        def instantiate(self, **kwargs):
            self.calls.append(kwargs)

    class Recording:
        checked = True

        def __init__(self):
            self.calls = []

        def instantiate(self, **kwargs):
            self.calls.append(kwargs)

    stimulus = Stimulus()
    recording = Recording()
    protocol = BPEMProtocol.__new__(BPEMProtocol)
    protocol.stimuli = [stimulus]
    protocol.recordings = [recording]
    cell_model = SimpleNamespace(icell="icell")

    protocol.instantiate(sim="sim", cell_model=cell_model)

    assert stimulus.calls == [{"sim": "sim", "icell": "icell"}]
    assert recording.calls == [{"sim": "sim", "icell": "icell"}]


def test_bpem_protocol_instantiate_checks_unchecked_recordings(monkeypatch):
    recording = SimpleNamespace(checked=False, instantiate=lambda **kwargs: None)
    stimulus = SimpleNamespace(instantiate=lambda **kwargs: None)
    protocol = BPEMProtocol.__new__(BPEMProtocol)
    protocol.stimuli = [stimulus]
    protocol.recordings = [recording]
    checked = []

    def check_recordings(recordings, icell, sim):
        checked.append((recordings, icell, sim))
        return recordings

    monkeypatch.setattr(protocols_module, "check_recordings", check_recordings)
    protocol.instantiate(sim="sim", cell_model=SimpleNamespace(icell="icell"))

    assert checked == [([recording], "icell", "sim")]


def test_rmp_protocol_run_calculates_target(monkeypatch):
    target = SimpleNamespace(calculate_feature=lambda response: [-72.5])
    protocol = RMPProtocol.__new__(RMPProtocol)
    protocol.recording_name = "RMP.soma.v"
    protocol.output_key = "bpo_rmp"
    protocol.target_voltage = target
    monkeypatch.setattr(
        protocols_module.BPEMProtocol,
        "run",
        lambda *args, **kwargs: {"RMP.soma.v": "trace"},
    )

    result = protocol.run(None, responses={})

    assert result == {"RMP.soma.v": "trace", "bpo_rmp": -72.5}


def test_rmp_protocol_run_returns_none_for_missing_trace(monkeypatch):
    protocol = RMPProtocol.__new__(RMPProtocol)
    protocol.recording_name = "RMP.soma.v"
    protocol.output_key = "bpo_rmp"
    protocol.target_voltage = SimpleNamespace(calculate_feature=lambda _: [-70.0])
    monkeypatch.setattr(
        protocols_module.BPEMProtocol,
        "run",
        lambda *a, **k: {"RMP.soma.v": None},
    )

    assert protocol.run(None, responses={}) == {
        "RMP.soma.v": None,
        "bpo_rmp": None,
    }


def test_rin_protocol_run_sets_holding_dependency(monkeypatch):
    protocol = RinProtocol.__new__(RinProtocol)
    protocol.dependencies = {
        "stimulus.holding_current": ["hold", "bpo_holding_current"]
    }
    protocol.stimulus = SimpleNamespace(holding_current=None)
    protocol.recording_name = "Rin.soma.v"
    protocol.output_key = "bpo_rin"
    protocol.target_rin = SimpleNamespace(calculate_feature=lambda _: [120.0])
    monkeypatch.setattr(
        protocols_module.BPEMProtocol,
        "run",
        lambda *args, **kwargs: {"Rin.soma.v": "trace"},
    )

    result = protocol.run(None, responses={"bpo_holding_current": -0.05})

    assert protocol.stimulus.holding_current == -0.05
    assert result == {"Rin.soma.v": "trace", "bpo_rin": 120.0}


def test_holding_get_voltage_base_rejects_spikes(monkeypatch):
    protocol = SearchHoldingCurrent.__new__(SearchHoldingCurrent)
    protocol.stimuli = [SimpleNamespace(amp=None)]
    protocol.recording_name = "SearchHoldingCurrent.soma.v"
    protocol.no_spikes = True
    protocol.spike_feature = SimpleNamespace(calculate_feature=lambda _: 1.0)
    protocol.target_voltage = SimpleNamespace(calculate_feature=lambda _: [-70.0])
    monkeypatch.setattr(
        protocols_module.BPEMProtocol,
        "run",
        lambda *args, **kwargs: {"SearchHoldingCurrent.soma.v": "trace"},
    )

    assert protocol.get_voltage_base(-0.1, None, {}, None, None) is None
    assert protocol.stimuli[0].amp == -0.1


def test_threshold_get_spikecount_handles_timeout_and_feature(monkeypatch):
    protocol = SearchThresholdCurrent.__new__(SearchThresholdCurrent)
    protocol.stimulus = SimpleNamespace(amp=None)
    protocol.recording_name = "SearchThresholdCurrent.soma.v"
    protocol.spikecount_timeout = 50
    protocol.spike_feature = SimpleNamespace(calculate_feature=lambda _: 3)
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return {"SearchThresholdCurrent.soma.v": "trace"}

    protocol._run = run
    assert protocol._get_spikecount(0.2, None, {}, None, None) == 3
    assert protocol.stimulus.amp == 0.2
    protocol._run = lambda *args, **kwargs: {"SearchThresholdCurrent.soma.v": None}
    assert protocol._get_spikecount(0.3, None, {}, None, None) == 2
    assert calls[0]["timeout"] == 50
