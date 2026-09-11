"""Tests for protocol dependency helpers."""

from types import SimpleNamespace

from bluepyemodel.evaluation.protocols import (
    NoHoldingCurrent,
    ProtocolRunner,
    ResponseDependencies,
    SearchHoldingCurrent,
    SearchThresholdCurrent,
)


def test_response_dependencies_set_attributes_and_missing_values():
    dependency = ResponseDependencies(
        {
            "value": ["source", "source_value"],
            "nested.value": ["source", "nested_value"],
        }
    )
    dependency.nested = SimpleNamespace(value=None)

    assert (
        dependency.set_dependencies({"source_value": 1.5, "nested_value": 2.5}) is True
    )
    assert dependency.value == 1.5
    assert dependency.nested.value == 2.5
    assert dependency.set_dependencies({}) is False


def test_no_holding_current_returns_zero():
    protocol = NoHoldingCurrent("SearchHoldingCurrent")

    assert protocol.run(cell_model=None) == {"bpo_holding_current": 0}


class FakeProtocol:
    def __init__(self, name, result, dependencies=None, output_key=None):
        self.name = name
        self.result = result
        self.dependencies = dependencies or {}
        self.output_key = output_key
        self.calls = []

    def run(self, cell_model, param_values, sim, isolate, timeout, responses):
        self.calls.append((cell_model, param_values, responses.copy()))
        return self.result


def test_protocol_runner_orders_dependencies_and_passes_responses():
    source = FakeProtocol("source", {"source_key": 1})
    dependent = FakeProtocol(
        "dependent",
        {"dependent_key": 2},
        dependencies={"value": ["source", "source_key"]},
    )
    runner = ProtocolRunner({"dependent": dependent, "source": source})
    cell_model = SimpleNamespace(frozen=[], unfrozen=[])
    cell_model.freeze = lambda values: cell_model.frozen.append(values)
    cell_model.unfreeze = lambda keys: cell_model.unfrozen.append(list(keys))

    responses = runner.run(cell_model, {"parameter": 1})

    assert runner.execution_order == ["source", "dependent"]
    assert list(responses) == ["source_key", "dependent_key"]
    assert dependent.calls[0][1] == {}
    assert dependent.calls[0][2] == {"source_key": 1}
    assert cell_model.frozen == [{"parameter": 1}]
    assert cell_model.unfrozen == [["parameter"]]


def test_protocol_runner_stops_when_protocol_returns_none_response():
    first = FakeProtocol("first", {"first_key": None})
    second = FakeProtocol("second", {"second_key": 2})
    runner = ProtocolRunner({"first": first, "second": second})
    cell_model = SimpleNamespace(freeze=lambda values: None, unfreeze=lambda keys: None)

    responses = runner.run(cell_model, {})

    assert responses == {}
    assert second.calls == []


def test_search_holding_current_bisection_converges_at_midpoint():
    search = SearchHoldingCurrent.__new__(SearchHoldingCurrent)
    search.holding_voltage = -70.0
    search.voltage_precision = 0.1
    search.max_depth = 7
    search.get_voltage_base = lambda **kwargs: -70.0

    result = search.bisection_search(
        None, {}, None, None, upper_bound=0.2, lower_bound=-0.2
    )

    assert result == 0.0


def test_search_holding_current_returns_lower_bound_at_max_depth():
    search = SearchHoldingCurrent.__new__(SearchHoldingCurrent)
    search.holding_voltage = -70.0
    search.voltage_precision = 0.01
    search.max_depth = -1
    search.get_voltage_base = lambda **kwargs: -60.0

    result = search.bisection_search(
        None, {}, None, None, upper_bound=0.2, lower_bound=-0.2
    )

    assert result == -0.2


def test_search_threshold_current_defines_valid_bounds():
    search = SearchThresholdCurrent.__new__(SearchThresholdCurrent)
    search.max_threshold_voltage = -30.0
    search.rmp = -70.0
    search.rin = 100.0
    search.hold_key = "holding"
    search.no_spikes = True
    search._get_spikecount = lambda current, *args: int(current > 0.0)

    bounds = search.define_search_bounds(None, {}, None, None, {"holding": -0.1})

    assert bounds == (-0.1, 0.4)


def test_search_threshold_current_rejects_spikes_at_lower_bound():
    search = SearchThresholdCurrent.__new__(SearchThresholdCurrent)
    search.max_threshold_voltage = -30.0
    search.rmp = -70.0
    search.rin = 100.0
    search.hold_key = "holding"
    search.no_spikes = True
    search._get_spikecount = lambda current, *args: 1

    assert search.define_search_bounds(None, {}, None, None, {"holding": -0.1}) == (
        None,
        None,
    )


def test_search_threshold_current_bisection_stops_at_precision():
    search = SearchThresholdCurrent.__new__(SearchThresholdCurrent)
    search.current_precision = 2.0
    search.max_depth = 5
    search._get_spikecount = lambda current, *args: 0

    result = search.bisection_search(
        None, {}, None, None, upper_bound=1.0, lower_bound=0.0
    )

    assert result == 1.0


def test_search_threshold_current_caps_maximum_current():
    search = SearchThresholdCurrent.__new__(SearchThresholdCurrent)
    search.max_threshold_voltage = 100.0
    search.rmp = -70.0
    search.rin = 1.0

    assert search.max_threshold_current() == 2.0
