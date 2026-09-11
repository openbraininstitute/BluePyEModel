"""Tests for protocol dependency helpers."""

from types import SimpleNamespace

from bluepyemodel.evaluation.protocols import NoHoldingCurrent, ResponseDependencies


def test_response_dependencies_set_attributes_and_missing_values():
    dependency = ResponseDependencies(
        {"value": ["source", "source_value"], "nested.value": ["source", "nested_value"]}
    )
    dependency.nested = SimpleNamespace(value=None)

    assert dependency.set_dependencies({"source_value": 1.5, "nested_value": 2.5}) is True
    assert dependency.value == 1.5
    assert dependency.nested.value == 2.5
    assert dependency.set_dependencies({}) is False


def test_no_holding_current_returns_zero():
    protocol = NoHoldingCurrent("SearchHoldingCurrent")

    assert protocol.run(cell_model=None) == {"bpo_holding_current": 0}
