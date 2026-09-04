"""Tests for distance-dependent distribution schemas."""

import pytest

from bluepyemodel.preprocessing.distributions import resolve_distance_dependent_distribution
from bluepyemodel.preprocessing.schemas import (
    STANDARD_DISTANCE_DEPENDENT_DISTRIBUTIONS,
    CustomDistanceDependentDistribution,
    DistanceDependentDistribution,
    ExponentialDistanceDependentDistribution,
    LinearHDPasDistanceDependentDistribution,
    StepDistanceDependentDistribution,
    UniformDistanceDependentDistribution,
)


@pytest.mark.parametrize(
    ("distribution_class", "expected_name", "expected_function"),
    [
        (UniformDistanceDependentDistribution, "uniform", None),
        (
            ExponentialDistanceDependentDistribution,
            "exp",
            "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}",
        ),
        (
            StepDistanceDependentDistribution,
            "step",
            (
                "{value} * (0.1 + 0.9 * int(({distance} > {step_begin}) & "
                "({distance} < {step_end})))"
            ),
        ),
        (
            LinearHDPasDistanceDependentDistribution,
            "linear_hdpas",
            "(1. + 3./100. * {distance})*{value}",
        ),
    ],
)
def test_legacy_distributions_serialize(distribution_class, expected_name, expected_function):
    distribution = distribution_class()

    assert distribution.function == expected_function
    assert distribution.to_emc_dict() == {
        "name": expected_name,
        "function": expected_function,
        "soma_ref_location": 0.5,
    }


def test_custom_distribution_is_validated_and_serialized():
    distribution = CustomDistanceDependentDistribution(
        name="custom_profile",
        function="({value} + {distance}) / 2",
        soma_ref_location=0.25,
    )

    assert distribution.function == "({value} + {distance}) / 2"
    assert distribution.to_emc_dict() == {
        "name": "custom_profile",
        "function": "({value} + {distance}) / 2",
        "soma_ref_location": 0.25,
    }


def test_custom_distribution_requires_declared_parameters():
    with pytest.raises(ValueError, match=r"\{constant\} placeholder"):
        CustomDistanceDependentDistribution(
            name="custom",
            function="math.exp({distance})*{value}",
            parameters=["constant"],
        )


def test_distribution_parameters_require_a_function():
    with pytest.raises(ValueError, match="must define a function"):
        DistanceDependentDistribution(parameters=["constant"])


def test_step_distribution_preserves_morphology_derived_placeholders():
    distribution = StepDistanceDependentDistribution()

    assert "{step_begin}" in distribution.function
    assert "{step_end}" in distribution.function
    assert distribution.parameters is None


def test_standard_distributions_are_available_without_declaration():
    assert set(STANDARD_DISTANCE_DEPENDENT_DISTRIBUTIONS) == {
        "uniform",
        "exp",
        "step",
        "exp_na_dend",
        "linear_hd_apic",
        "sigmoid_kad_apic",
        "linear_e_pas_apic",
        "linear_hdpas",
        "sigmoid_kad",
        "sigmoid_kdbm_apic",
    }
    resolved = resolve_distance_dependent_distribution("linear_hdpas", {})
    assert isinstance(resolved, LinearHDPasDistanceDependentDistribution)
    assert resolved.function == "(1. + 3./100. * {distance})*{value}"


def test_resolve_returns_custom_distribution_when_not_standard():
    custom = CustomDistanceDependentDistribution(
        name="mouse_decay",
        function="math.exp({distance})*{value}",
    )
    resolved = resolve_distance_dependent_distribution("mouse_decay", {"mouse_decay": custom})
    assert resolved is custom
    assert resolve_distance_dependent_distribution("missing", {}) is None
