"""Distribution catalog helpers for preprocessing."""

from collections.abc import Mapping

from bluepyemodel.preprocessing.schemas import STANDARD_DISTANCE_DEPENDENT_DISTRIBUTIONS
from bluepyemodel.preprocessing.schemas import CustomDistanceDependentDistribution
from bluepyemodel.preprocessing.schemas import DistanceDependentDistributionUnion


def resolve_distance_dependent_distribution(
    name: str,
    custom_distributions: Mapping[str, CustomDistanceDependentDistribution],
) -> DistanceDependentDistributionUnion | None:
    """Resolve a distribution name against the standard catalog, then custom declarations."""
    standard = STANDARD_DISTANCE_DEPENDENT_DISTRIBUTIONS.get(name)
    if standard is not None:
        return standard
    return custom_distributions.get(name)
