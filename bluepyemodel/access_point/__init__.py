"""E-model access_point module"""

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


NEXUS_REMOVED_MESSAGE = (
    "The Nexus access point was removed in BluePyEModel 2.0, because no Nexus deployment with "
    "the data it expects exists any more. Use the 'local' access point, or BluePyEModel 1.x "
    "for the Nexus access point."
)


def get_access_point(access_point, emodel, **kwargs):
    """Returns a DataAccessPoint object.

    Args:
        access_point (str): name of the access_point to use. Only 'local' is supported.
        emodel (str): name of the emodel.
        kwargs: extra arguments to pass to the access point constructor, see below.

    Optional:
        - etype (str): name of the electric type.
        - ttype (str): name of the transcriptomic type.
        - mtype (str): name of the morphology type.
        - species (str): name of the species.
        - brain_region (str): name of the brain location.
        - iteration_tag (str): tag associated to the current run.
        - synapse_class (str): synapse class (neurotransmitter).
        - emodel_dir (str): path of the directory containing the parameters,
          features and parameters config files.
        - recipes_path (str): path to the file containing the recipes.
        - final_path (str): path to the final.json, if different from the one in emodel_dir.
        - legacy_dir_structure (bool): uses legacy folder structure.
        - with_seeds (bool): allows for emodel_seed type of emodel names in final.json
          (not in recipes).

    Returns:
        LocalAccessPoint

    Raises:
        ValueError: if access_point is not 'local'. The Nexus access point was removed in 2.0.
    """

    if access_point == "nexus":
        raise ValueError(NEXUS_REMOVED_MESSAGE)
    if access_point != "local":
        raise ValueError(f"Unknown access point: {access_point}. Should be 'local'.")

    etype = kwargs.get("etype", None)
    etype = etype.replace("__", " ") if etype else None

    ttype = kwargs.get("ttype", None)
    ttype = ttype.replace("__", " ") if ttype else None

    mtype = kwargs.get("mtype", None)
    mtype = mtype.replace("__", " ") if mtype else None

    brain_region = kwargs.get("brain_region", None)
    brain_region = brain_region.replace("__", " ") if brain_region else None

    from bluepyemodel.access_point.local import LocalAccessPoint

    return LocalAccessPoint(
        emodel=emodel,
        emodel_dir=kwargs.get("emodel_dir", None),
        etype=etype,
        ttype=ttype,
        mtype=mtype,
        species=kwargs.get("species", None),
        brain_region=brain_region,
        iteration_tag=kwargs.get("iteration_tag", None),
        synapse_class=kwargs.get("synapse_class", None),
        recipes_path=kwargs.get("recipes_path", None),
        final_path=kwargs.get("final_path", None),
        legacy_dir_structure=kwargs.get("legacy_dir_structure", False),
        with_seeds=kwargs.get("with_seeds", False),
    )
