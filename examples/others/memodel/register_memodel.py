"""Register an memodel"""

import copy
import getpass

from kgforge.core import KnowledgeGraphForge

from bluepyemodel.access_point.forge_access_point import get_brain_region_notation
from bluepyemodel.access_point.nexus import NexusAccessPoint
from bluepyemodel.emodel_pipeline.memodel import MEModel
from bluepyemodel.emodel_pipeline.plotting import plot_models


def connect_forge(bucket, endpoint, access_token, forge_path=None):
    """Creation of a forge session"""
    if not forge_path:
        forge_path = (
            "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
            + "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
        )
    forge = KnowledgeGraphForge(forge_path, bucket=bucket, endpoint=endpoint, token=access_token)
    return forge


def get_morph_mtype(annotation):
    morph_mtype = None
    if hasattr(annotation, "hasBody"):
        if hasattr(annotation.hasBody, "label"):
            morph_mtype = annotation.hasBody.label
        else:
            raise ValueError("Morphology resource has no label in annotation.hasBody.")
    else:
        raise ValueError("Morphology resource has no hasBodz in annotation.")

    return morph_mtype


def get_morph_metadata(access_point, morph_id):
    resource = access_point.access_point.retrieve(morph_id)
    if resource is None:
        raise TypeError(f"Could not find the morphology resource with id {morph_id}")

    morph_brain_region = None
    if hasattr(resource, "brainLocation"):
        if hasattr(resource.brainLocation, "brainRegion"):
            if hasattr(resource.brainLocation.brainRegion, "label"):
                morph_brain_region = resource.brainLocation.brainRegion.label
            else:
                raise AttributeError(
                    "Morphology resource has no label in brainLocation.brainRegion"
                )
        else:
            raise AttributeError("Morphology resource has no brainRegion in brainLocation.")
    else:
        raise AttributeError("Morphology resource has no brainLocation.")

    morph_mtype = None
    if not hasattr(resource, "annotation"):
        raise AttributeError("Morphology resource has no annotation.")

    if isinstance(resource.annotation, dict):
        if hasattr(resource.annotation, "type") and (
            "MTypeAnnotation" in resource.annotation.type
            or "nsg:MTypeAnnotation" in resource.annotation.type
        ):
            morph_mtype = get_morph_mtype(resource.annotation)
    elif isinstance(resource.annotation, list):
        for annotation in resource.annotation:
            if hasattr(annotation, "type") and (
                "MTypeAnnotation" in annotation.type or "nsg:MTypeAnnotation" in annotation.type
            ):
                morph_mtype = get_morph_mtype(annotation)

    if morph_mtype is None:
        raise TypeError("Could not find mtype in morphology resource")

    return morph_mtype, morph_brain_region


def get_new_emodel_metadata(
    access_point,
    morph_id,
    morph_name,
    update_emodel_name,
    use_brain_region_from_morphology,
    use_mtype_in_githash,
):
    new_emodel_metadata = copy.deepcopy(access_point.emodel_metadata)
    new_mtype, new_br = get_morph_metadata(access_point, morph_id)
    new_emodel_metadata.mtype = new_mtype

    if update_emodel_name:
        new_emodel_metadata.emodel = f"{new_emodel_metadata.etype}_{new_mtype}"

    if use_brain_region_from_morphology:
        new_emodel_metadata.brain_region = new_br
        new_emodel_metadata.allen_notation = get_brain_region_notation(
            new_br,
            access_point.access_point.access_token,
            access_point.forge_ontology_path,
        )

    if use_mtype_in_githash:
        new_emodel_metadata.iteration = f"{new_emodel_metadata.iteration}-{morph_name}"

    return new_emodel_metadata


def plot(access_point, seed, cell_evaluator, figures_dir, mapper):
    """Plot figures and return total fitness (sum of scores), holding and threshold currents"""
    # compute scores
    # we need to do this outside of main plotting function with custom function
    # so that we do not take old emodel scores in scores figure
    emodel_score = plot_scores(access_point, cell_evaluator, mapper, figures_dir, seed)

    emodels = plot_models(
        access_point=access_point,
        mapper=mapper,
        seeds=[seed],
        figures_dir=figures_dir,
        plot_distributions=True,
        plot_scores=False,  # scores figure done outside of this
        plot_traces=True,
        plot_thumbnail=True,
        plot_currentscape=access_point.pipeline_settings.plot_currentscape,
        plot_bAP_EPSP=access_point.pipeline_settings.plot_bAP_EPSP,
        plot_dendritic_ISI_CV=True,  # for detailed cADpyr cells. will be skipped otherwise
        plot_dendritic_rheobase=True,  # for detailed cADpyr cells. will be skipped otherwise
        only_validated=False,
        save_recordings=False,
        load_from_local=False,
        cell_evaluator=cell_evaluator,  # <-- feed the modified evaluator here
    )
    emodel_holding = emodels[0].responses.get("bpo_holding_current", None)
    emodel_threshold = emodels[0].responses.get("bpo_threshold_current", None)

    return emodel_score, emodel_holding, emodel_threshold


if __name__ == "__main__":
    project = "mmb-point-neuron-framework-model"  # replace with a valid Nexus project name
    organisation = "bbp"  # replace with the organisation name
    endpoint = "https://openbluebrain.com/api/nexus/v1"  # replace with the Nexus endpoint url
    forge_path = "./forge.yml"  # this file has to be present
    forge_ontology_path = "./forge_ontology_path.yml"  # this file also
    # memodel_id = "<MEMODEL ID>" # replace with the id of the MEModel you want to update

    mapper = map

    # MEModel metadata-related config
    update_emodel_name = True
    use_brain_region_from_morphology = True
    use_mtype_in_githash = True  # to distinguish from other MEModel

    # create forge and retrieve ME-Model
    access_token = getpass.getpass()
    forge = connect_forge(
        bucket=f"{organisation}/{project}",
        endpoint=endpoint,
        access_token=access_token,
        forge_path=forge_path,
    )

    # memodel resource
    # memodel_r = forge.retrieve(memodel_id)
    # emodel_id, morph_id = get_ids_from_memodel(memodel_r)
    emodel_id = "<EMODEL ID>"
    morph_id = "<MORPH ID>"
    emodel_r = forge.retrieve(emodel_id)
    morph_r = forge.retrieve(morph_id)

    # get metadata from EModel resource
    emodel = emodel_r.eModel if hasattr(emodel_r, "eModel") else None
    etype = emodel_r.eType if hasattr(emodel_r, "eType") else None
    ttype = emodel_r.tType if hasattr(emodel_r, "tType") else None
    mtype = emodel_r.mType if hasattr(emodel_r, "mType") else None
    species = None
    if hasattr(emodel_r, "subject"):
        if hasattr(emodel_r.subject, "species"):
            species = (
                emodel_r.subject.species.label
                if hasattr(emodel_r.subject.species, "label")
                else None
            )
    brain_region = None
    if hasattr(emodel_r, "brainLocation"):
        if hasattr(emodel_r.brainLocation, "brainRegion"):
            brain_region = (
                emodel_r.brainLocation.brainRegion.label
                if hasattr(emodel_r.brainLocation.brainRegion, "label")
                else None
            )
    iteration_tag = emodel_r.iteration if hasattr(emodel_r, "iteration") else None
    synapse_class = emodel_r.synapse_class if hasattr(emodel_r, "synapseClass") else None
    seed = int(emodel_r.seed if hasattr(emodel_r, "seed") else 0)

    # get morph metadata
    morph_name = morph_r.name if hasattr(morph_r, "name") else None
    morph_format = "swc"  # assumes swc is always present and we do not care about small differences between format

    # additional metadata we will need when saving me-model resource
    subject_ontology = emodel_r.subject if hasattr(emodel_r, "subject") else None
    brain_location_ontology = morph_r.brainLocation if hasattr(morph_r, "brainLocation") else None

    # feed nexus acces point with appropriate data
    access_point = NexusAccessPoint(
        emodel=emodel,
        etype=etype,
        ttype=ttype,
        mtype=mtype,
        species=species,
        brain_region=brain_region,
        iteration_tag=iteration_tag,
        synapse_class=synapse_class,
        project=project,
        organisation=organisation,
        endpoint=endpoint,
        forge_path=forge_path,  # this file has to be present
        forge_ontology_path=forge_ontology_path,  # this file also
        access_token=access_token,
    )

    # update settings for better threshold precision
    access_point.pipeline_settings.current_precision = 2e-3

    # get cell evaluator with 'new' morphology
    cell_evaluator = get_cell_evaluator(access_point, morph_name, morph_format, morph_id)

    # get new emodel metadata (mtype, emodel, brain region, iteration/githash)
    # to correspond to combined metadata of emodel and morphology
    new_emodel_metadata = get_new_emodel_metadata(
        access_point,
        morph_id,
        morph_name,
        update_emodel_name,
        use_brain_region_from_morphology,
        use_mtype_in_githash,
    )

    # create MEModel
    memodel = MEModel(
        seed=seed,
        emodel_metadata=new_emodel_metadata,
        emodel_id=emodel_id,
        morphology_id=morph_id,
        validated=False,
    )

    def store_memodel(access_point, memodel, description=None):
        """Store an MEModel on Nexus"""

        access_point.store_object(
            memodel,
            seed=memodel.seed,
            description=description,
            is_analysis_suitable=True,
        )
        # wait for the object to be uploaded and fetchable
        time.sleep(access_point.sleep_time)
