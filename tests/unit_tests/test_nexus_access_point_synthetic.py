"""Tests of ``NexusAccessPoint`` against an in-memory Nexus. No network or credentials."""

import os
import pathlib

import pytest
from kgforge.core import Resource

from bluepyemodel.access_point import forge_access_point as fap
from bluepyemodel.access_point import nexus as nexus_module
from bluepyemodel.access_point.forge_access_point import AccessPointException
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint
from bluepyemodel.access_point.nexus import NexusAccessPoint
from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration
from bluepyemodel.emodel_pipeline.emodel import EModel
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.emodel_pipeline.emodel_workflow import EModelWorkflow
from bluepyemodel.evaluation.fitness_calculator_configuration import FitnessCalculatorConfiguration
from bluepyemodel.model.distribution_configuration import DistributionConfiguration
from bluepyemodel.model.mechanism_configuration import MechanismConfiguration
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration
from tests.unit_tests.nexus_fakes import FakeNexus
from tests.unit_tests.nexus_fakes import make_token
from tests.unit_tests.nexus_fakes import to_json

ATLAS_ID = "https://bbp.epfl.ch/neurosciencegraph/data/4906ab85-694f-469d-962f-c0174e901885"
REGION = "somatosensory areas"
MOUSE = "Mus musculus"
WORKFLOW = {
    "targets_configuration_id": "t",
    "pipeline_settings_id": "p",
    "emodel_configuration_id": "c",
}
HOLDING_KEY = "SearchHoldingCurrent.soma.v.bpo_holding_current"
THRESHOLD_KEY = "SearchThresholdCurrent.soma.v.bpo_threshold_current"


@pytest.fixture
def nexus(monkeypatch):
    world = FakeNexus()
    world.ontology.ontology[REGION] = Resource(id="br:ss", label=REGION, notation="SS")
    world.atlas.add("AtlasRelease", id_=ATLAS_ID)
    monkeypatch.setattr(fap, "KnowledgeGraphForge", world.connect)
    monkeypatch.setattr(NexusForgeAccessPoint, "forges", {})
    return world


@pytest.fixture
def ap(nexus, workspace):
    return NexusAccessPoint(
        emodel="L5_TPC",
        etype="cAC",
        ttype="t1",
        mtype="L5_TPC:B",
        species="mouse",
        brain_region="SSCX",
        iteration_tag="v0",
        project="emodel",
        organisation="demo",
        access_token=make_token(),
        sleep_time=0,
    )


@pytest.fixture
def forge(nexus):
    return nexus.project


@pytest.fixture
def ontology_ap(nexus):
    return fap.ontology_forge_access_point(make_token())


def identity(ap, **extra):
    """Return the metadata a resource needs in order to belong to the e-model of ``ap``."""
    return {**ap.emodel_metadata_ontology.filters_for_resource(), **extra}


def make_emodel(ap, seed=1, passed=True, features=None, fitness=2.0):
    return EModel(
        fitness=fitness,
        parameter={"g": 1.0},
        score={"a": fitness},
        features=features or {"f": 0.1},
        scoreValidation={},
        passedValidation=passed,
        seed=seed,
        emodel_metadata=ap.emodel_metadata,
    )


def add_emodel(ap, forge, **kwargs):
    emodel = make_emodel(ap, **kwargs)
    return forge.add("EModel", payload=emodel.as_dict(), **identity(ap, seed=emodel.seed))


def add_workflow(ap, forge, **payload):
    return forge.add("EModelWorkflow", payload={**WORKFLOW, **payload}, **identity(ap))


def add_mechanism(forge, temperature=None, ljp=None, species=MOUSE, region=REGION, **fields):
    extra = dict(fields)
    if temperature is not None:
        extra["temperature"] = temperature
    if ljp is not None:
        extra["isLjpCorrected"] = ljp
    return forge.add(
        "SubCellularModelScript",
        name="NaTg",
        subject={"species": {"label": species}},
        brainLocation={"brainRegion": {"label": region}},
        **extra,
    )


def make_model_configuration(mechanisms=("pas", "NaTg"), morphology=None):
    return NeuronModelConfiguration(
        parameters=[],
        mechanisms=[{"name": name, "location": "all"} for name in mechanisms],
        distributions=[],
        morphology=morphology or {"name": "m1", "format": "asc"},
    )


def make_fcc_payload(**overrides):
    protocol = {
        "name": "Step",
        "stimuli": [{"amp": 0.1}],
        "recordings": [
            {"type": "CompRecording", "name": "Step.soma.v", "location": "soma", "variable": "v"}
        ],
    }
    payload = FitnessCalculatorConfiguration(efeatures=[], protocols=[protocol]).as_dict()
    payload.update(overrides)
    return payload


@pytest.fixture
def compile_calls(monkeypatch):
    calls = []

    def fake_run(command, **_):
        calls.append((command, os.getcwd()))

    monkeypatch.setattr(nexus_module.subprocess, "run", fake_run)
    return calls


# Construction


def test_init_resolves_ontology_metadata(ap):
    assert ap.emodel_metadata.allen_notation == "SS"
    assert ap.emodel_metadata_ontology.species["species"]["label"] == MOUSE
    assert ap.emodel_metadata_ontology.brain_region == {
        "type": "BrainLocation",
        "brainRegion": {"id": "br:ss", "label": REGION},
    }
    assert isinstance(ap.pipeline_settings, EModelPipelineSettings)
    assert ap.download_directory.is_dir()


def test_init_uses_explicit_ontology_forge_path(nexus, workspace):
    ap = NexusAccessPoint(
        emodel="L5_TPC",
        species="mouse",
        brain_region="SSCX",
        iteration_tag="v0",
        forge_path="data.yml",
        forge_ontology_path="ontology.yml",
        access_token=make_token(),
    )
    assert ap.forge_ontology_path == "ontology.yml"
    assert {c.forge_path for c in nexus.connections} == {"data.yml", "ontology.yml"}


def test_download_directory_is_per_iteration(ap):
    assert ap.download_directory == pathlib.Path("./nexus_temp/v0")
    assert ap.get_mechanisms_directory() == pathlib.Path("nexus_temp/v0/mechanisms").resolve()


# Checking e-, m- and t-types


def seed_types(nexus):
    answers = {
        "EType": "e_type_id",
        "MType": "m_type_id",
        "BrainCellTranscriptomeType": "t_type_id",
    }
    nexus.ontology.sparql_answers = [
        ((marker,), [Resource(**{field: f"{marker}:1"})]) for marker, field in answers.items()
    ]
    nexus.ontology.ontology.update(
        {
            "cAC": Resource(id="EType:1"),
            "L5_TPC:B": Resource(id="MType:1"),
            "t1": Resource(id="BrainCellTranscriptomeType:1"),
        }
    )


def test_check_mettypes_accepts_known_types(ap, nexus):
    seed_types(nexus)
    ap.check_mettypes()
    checked = [call["text"] for call in nexus.ontology.resolve_calls]
    assert checked[-3:] == ["cAC", "L5_TPC:B", "t1"]


def test_check_mettypes_skips_absent_types(ap, nexus):
    seed_types(nexus)
    ap.emodel_metadata.mtype = None
    ap.emodel_metadata.ttype = None
    ap.check_mettypes()
    assert nexus.ontology.resolve_calls[-1]["text"] == "cAC"


@pytest.mark.parametrize("unknown", ["cAC", "L5_TPC:B", "t1"])
def test_check_mettypes_rejects_unknown_type(ap, nexus, unknown):
    seed_types(nexus)
    del nexus.ontology.ontology[unknown]
    with pytest.raises(AccessPointException, match="Could not find"):
        ap.check_mettypes()


# Pipeline settings and generic storage


def test_pipeline_settings_round_trip(ap, forge):
    ap.store_pipeline_settings(EModelPipelineSettings(n_model=7))
    assert ap.get_pipeline_settings(strict=True).n_model == 7
    assert ap.has_pipeline_settings() is True


def test_pipeline_settings_missing(ap):
    assert ap.has_pipeline_settings() is False
    with pytest.raises(AccessPointException):
        ap.get_pipeline_settings(strict=True)
    assert ap.get_pipeline_settings(strict=False).n_model == 3


def test_pipeline_settings_are_found_through_legacy_metadata(ap, forge):
    legacy = ap.emodel_metadata_ontology.filters_for_resource_legacy()
    forge.add("EModelPipelineSettings", payload={"n_model": 9}, **legacy)
    assert ap.get_pipeline_settings(strict=True).n_model == 9


def test_store_object_adds_optional_metadata(ap, forge):
    ap.store_object(EModelWorkflow("t", "p", "c"), seed=4, description="note")
    stored = forge.active("EModelWorkflow")[0]
    assert (stored.seed, stored.description) == (4, "note")


def test_store_object_without_optional_metadata(ap, forge):
    ap.store_object(EModelWorkflow("t", "p", "c"))
    stored = forge.active("EModelWorkflow")[0]
    assert not hasattr(stored, "seed")
    assert not hasattr(stored, "description")


# Targets configuration


def nwb_trace(ap, nexus, name="c1", **fields):
    return nexus.traces.add(
        "ExperimentalTrace",
        name=name,
        files={f"{name}.nwb": b""},
        subject=ap.emodel_metadata_ontology.species,
        **fields,
    )


def targets_payload(files):
    return TargetsConfiguration(files=files, targets=[], protocols_rheobase=[]).as_dict()


def test_targets_configuration_resolves_trace_paths(ap, forge, nexus):
    trace = nwb_trace(ap, nexus)
    payload = targets_payload([{"cell_name": "c1", "id": trace.id, "ecodes": {}}])
    forge.add("ExtractionTargetsConfiguration", payload=payload, **identity(ap))
    configuration = ap.get_targets_configuration()
    assert configuration.files[0].filepath == str(ap.download_directory / "c1.nwb")
    assert [t.cell_name for t in configuration.available_traces] == ["c1"]
    assert "Spikecount" in configuration.available_efeatures


def test_targets_configuration_fills_empty_files_from_etype(ap, forge, nexus):
    annotation = {"name": "e-type annotation", "hasBody": {"label": "cAC"}}
    nwb_trace(ap, nexus, annotation=annotation)
    other = {"name": "e-type annotation", "hasBody": {"label": "bNAC"}}
    nwb_trace(ap, nexus, name="c2", annotation=other)
    payload = targets_payload([])
    forge.add("ExtractionTargetsConfiguration", payload=payload, **identity(ap))
    assert [f.cell_name for f in ap.get_targets_configuration().files] == ["c1"]


def test_targets_configuration_without_matching_trace(ap, forge):
    payload = targets_payload([])
    forge.add("ExtractionTargetsConfiguration", payload=payload, **identity(ap))
    with pytest.raises(AccessPointException, match="Could not find any trace"):
        ap.get_targets_configuration()


def test_has_targets_configuration(ap, forge, nexus):
    assert ap.has_targets_configuration() is False
    trace = nwb_trace(ap, nexus)
    payload = targets_payload([{"cell_name": "c1", "id": trace.id, "ecodes": {}}])
    forge.add("ExtractionTargetsConfiguration", payload=payload, **identity(ap))
    assert ap.has_targets_configuration() is True


def test_store_targets_configuration_links_known_traces(ap, forge, nexus, caplog):
    trace = nwb_trace(ap, nexus)
    configuration = TargetsConfiguration(
        files=[{"cell_name": "c1", "ecodes": {}}, {"cell_name": "unknown", "ecodes": {}}],
        targets=[],
        protocols_rheobase=[],
    )
    with caplog.at_level("WARNING", logger="__main__"):
        ap.store_targets_configuration(configuration)
    assert configuration.files[0].id == trace.id
    assert configuration.files[1].id is None
    assert "Trace unknown not found" in caplog.text
    assert forge.active("ExtractionTargetsConfiguration")


# Fitness calculator configuration


def test_fitness_configuration_uses_pipeline_defaults(ap, forge):
    ap.pipeline_settings = EModelPipelineSettings(
        name_rmp_protocol="R",
        name_Rin_protocol="I",
        validation_protocols=["V"],
        stochasticity=True,
    )
    payload = make_fcc_payload(stochasticity=None)
    forge.add("FitnessCalculatorConfiguration", payload=payload, **identity(ap))
    configuration = ap.get_fitness_calculator_configuration()
    assert configuration.name_rmp_protocol == "R"
    assert configuration.name_rin_protocol == "I"
    assert configuration.validation_protocols == ["V"]
    assert configuration.stochasticity is True


def test_fitness_configuration_keeps_its_own_values(ap, forge):
    ap.pipeline_settings = EModelPipelineSettings(name_rmp_protocol="R", stochasticity=True)
    payload = make_fcc_payload(
        name_rmp_protocol="own", validation_protocols=["own"], stochasticity=False
    )
    forge.add("FitnessCalculatorConfiguration", payload=payload, **identity(ap))
    configuration = ap.get_fitness_calculator_configuration()
    assert configuration.name_rmp_protocol == "own"
    assert configuration.validation_protocols == ["own"]
    assert configuration.stochasticity is False


def test_fitness_configuration_adds_ion_recordings(ap, forge, monkeypatch):
    forge.add("FitnessCalculatorConfiguration", payload=make_fcc_payload(), **identity(ap))
    monkeypatch.setattr(ap, "get_ion_currents_concentrations", lambda: (["ik"], ["ki"]))
    configuration = ap.get_fitness_calculator_configuration(record_ions_and_currents=True)
    variables = [r["variable"] for r in configuration.protocols[0].recordings]
    assert sorted(variables) == ["ik", "ki", "v"]


def test_fitness_configuration_without_known_ions(ap, forge, monkeypatch):
    forge.add("FitnessCalculatorConfiguration", payload=make_fcc_payload(), **identity(ap))
    monkeypatch.setattr(ap, "get_ion_currents_concentrations", lambda: (None, None))
    configuration = ap.get_fitness_calculator_configuration(record_ions_and_currents=True)
    assert [r["variable"] for r in configuration.protocols[0].recordings] == ["v"]


def test_store_fitness_configuration_requires_workflow(ap):
    with pytest.raises(AccessPointException, match="No EModelWorkflow"):
        ap.store_fitness_calculator_configuration(FitnessCalculatorConfiguration())


def test_store_fitness_configuration_links_workflow(ap, forge):
    workflow = add_workflow(ap, forge)
    configuration = FitnessCalculatorConfiguration(efeatures=[], protocols=[])
    ap.store_fitness_calculator_configuration(configuration)
    assert configuration.workflow_id == workflow.id
    stored = forge.active("FitnessCalculatorConfiguration")[0]
    assert forge.payload_of(forge.active("EModelWorkflow")[0])["fitness_configuration_id"] == (
        stored.id
    )


# Model configuration


def add_morphology(forge, id_="morph:1", name="m1", files=None, **fields):
    return forge.add(
        "NeuronMorphology", id_=id_, name=name, files=files or {"m1.asc": b""}, **fields
    )


def add_model_configuration(ap, forge, mechanisms=("pas",)):
    payload = make_model_configuration(
        mechanisms=mechanisms, morphology={"name": "m1", "format": "asc", "id": "morph:1"}
    ).as_dict()
    forge.add("EModelConfiguration", payload=payload, **identity(ap))


def test_get_model_configuration(ap, forge):
    add_morphology(forge)
    add_model_configuration(ap, forge)
    configuration = ap.get_model_configuration()
    assert configuration.morphology.path == str(ap.download_directory / "m1.asc")
    assert configuration.available_mechanisms is None
    assert not configuration.available_morphologies


def test_get_model_configuration_can_list_morphologies(ap, forge):
    add_morphology(forge)
    add_model_configuration(ap, forge)
    configuration = ap.get_model_configuration(skip_get_available_morph=False)
    assert configuration.available_morphologies == {"m1"}


def test_get_model_configuration_downloads_synapse_model(ap, forge, compile_calls):
    ap.pipeline_settings.use_ProbAMPANMDA_EMS = True
    forge.add("SynapsePhysiologyModel", name="ProbAMPANMDA_EMS", files={"syn.mod": b"x"})
    add_morphology(forge)
    add_model_configuration(ap, forge)
    ap.get_model_configuration()
    assert (ap.get_mechanisms_directory() / "ProbAMPANMDA_EMS.mod").is_file()
    assert [command for command, _ in compile_calls] == ["nrnivmodl mechanisms"]


def test_has_model_configuration(ap, forge):
    assert ap.has_model_configuration() is False
    add_morphology(forge)
    add_model_configuration(ap, forge)
    assert ap.has_model_configuration() is True


def test_has_fitness_configuration(ap, forge):
    assert ap.has_fitness_calculator_configuration() is False
    forge.add("FitnessCalculatorConfiguration", payload=make_fcc_payload(), **identity(ap))
    assert ap.has_fitness_calculator_configuration() is True


def test_store_model_configuration_requires_morphologies(ap):
    with pytest.raises(AccessPointException, match="Cannot find morphologies"):
        ap.store_model_configuration(make_model_configuration())


def test_store_model_configuration_sets_ids(ap, forge):
    add_morphology(forge)
    mechanism = add_mechanism(forge)
    configuration = make_model_configuration()
    ap.store_model_configuration(configuration)
    assert configuration.morphology.id == "morph:1"
    by_name = {m.name: m for m in configuration.mechanisms}
    assert by_name["NaTg"].id == mechanism.id
    assert by_name["pas"].id is None
    assert forge.active("EModelConfiguration")


def test_store_model_configuration_warns_on_unknown_morphology(ap, forge, caplog):
    add_morphology(forge, name="other")
    configuration = make_model_configuration(mechanisms=("pas",))
    with caplog.at_level("WARNING", logger="__main__"):
        ap.store_model_configuration(configuration)
    assert configuration.morphology.id is None
    assert "Morphology m1 not found" in caplog.text


def test_store_model_configuration_links_synapse_model(ap, forge):
    ap.pipeline_settings.use_ProbAMPANMDA_EMS = True
    add_morphology(forge)
    synapse = forge.add("SynapsePhysiologyModel", name="ProbAMPANMDA_EMS")
    configuration = make_model_configuration(mechanisms=("pas",))
    ap.store_model_configuration(configuration)
    assert configuration.extra_mech_ids == [(synapse.id, "SynapsePhysiologyModel")]


# Mechanism selection


def test_fetch_mechanism_not_found(ap, ontology_ap):
    with pytest.raises(AccessPointException, match="NaTg not found"):
        ap.fetch_and_filter_mechanism(MechanismConfiguration("NaTg", "all"), ontology_ap)


def test_fetch_mechanism_applies_every_filter(ap, forge, ontology_ap):
    add_mechanism(forge, temperature={"value": 20}, ljp=True)
    add_mechanism(forge, temperature={"value": 34}, ljp=False)
    add_mechanism(forge, temperature={"value": 34}, ljp=True, species="Homo sapiens")
    add_mechanism(forge, temperature={"value": 34}, ljp=True, region="other")
    wanted = add_mechanism(forge, temperature={"value": 34}, ljp=True)
    mechanism = MechanismConfiguration("NaTg", "all", temperature=34, ljp_corrected=True)
    assert ap.fetch_and_filter_mechanism(mechanism, ontology_ap) is wanted


def test_fetch_mechanism_accepts_plain_temperature(ap, forge, ontology_ap):
    add_mechanism(forge, temperature=20)
    wanted = add_mechanism(forge, temperature=34)
    mechanism = MechanismConfiguration("NaTg", "all", temperature=34)
    assert ap.fetch_and_filter_mechanism(mechanism, ontology_ap) is wanted


def test_fetch_mechanism_relaxes_filters_with_no_match(ap, forge, ontology_ap, caplog):
    first = add_mechanism(forge, temperature={"value": 20}, ljp=False, species="Rat", region="x")
    add_mechanism(forge, temperature={"value": 20}, ljp=False, species="Rat", region="x")
    mechanism = MechanismConfiguration("NaTg", "all", temperature=34, ljp_corrected=True)
    with caplog.at_level("WARNING", logger="__main__"):
        assert ap.fetch_and_filter_mechanism(mechanism, ontology_ap) is first
    assert "Keeping the 1st resource" in caplog.text


def test_fetch_mechanism_defaults_to_physiological_temperature(ap, forge, ontology_ap):
    add_mechanism(forge, temperature={"value": 20})
    wanted = add_mechanism(forge, temperature={"value": 34})
    mechanism = MechanismConfiguration("NaTg", "all")
    assert ap.fetch_and_filter_mechanism(mechanism, ontology_ap) is wanted


def test_fetch_mechanism_defaults_to_ljp_corrected(ap, forge, ontology_ap):
    add_mechanism(forge, ljp=False)
    wanted = add_mechanism(forge, ljp=True)
    mechanism = MechanismConfiguration("NaTg", "all")
    assert ap.fetch_and_filter_mechanism(mechanism, ontology_ap) is wanted


# Distributions


def test_distributions_round_trip(ap, forge):
    ap.store_distribution(DistributionConfiguration(name="linear", function="{value}*{distance}"))
    assert [d.name for d in ap.get_distributions()] == ["linear"]


# Workflow


def test_create_workflow_needs_only_the_model_configuration(ap, forge):
    configuration = forge.add("EModelConfiguration", **identity(ap))
    workflow = ap.create_emodel_workflow(state="running")
    assert workflow.emodel_configuration_id == configuration.id
    assert workflow.targets_configuration_id is None
    assert workflow.pipeline_settings_id is None
    assert workflow.fitness_configuration_id is None
    assert workflow.state == "running"


def test_create_workflow_collects_all_ids(ap, forge):
    ids = {
        type_: forge.add(type_, **identity(ap)).id
        for type_ in (
            "ExtractionTargetsConfiguration",
            "EModelPipelineSettings",
            "EModelConfiguration",
            "FitnessCalculatorConfiguration",
        )
    }
    workflow = ap.create_emodel_workflow()
    assert workflow.targets_configuration_id == ids["ExtractionTargetsConfiguration"]
    assert workflow.pipeline_settings_id == ids["EModelPipelineSettings"]
    assert workflow.fitness_configuration_id == ids["FitnessCalculatorConfiguration"]


def test_create_workflow_fails_without_model_configuration(ap):
    with pytest.raises(AccessPointException):
        ap.create_emodel_workflow()


def test_get_workflow(ap, forge):
    assert ap.get_emodel_workflow() == (None, None)
    stored = add_workflow(ap, forge)
    workflow, id_ = ap.get_emodel_workflow()
    assert id_ == stored.id
    assert workflow.targets_configuration_id == "t"


def test_check_workflow_configurations(ap, forge):
    configuration = forge.add("EModelConfiguration")
    assert ap.check_emodel_workflow_configurations(EModelWorkflow(None, None, configuration.id))
    assert not ap.check_emodel_workflow_configurations(EModelWorkflow("gone", None, None))


def test_store_workflow_when_absent(ap, forge):
    ap.store_or_update_emodel_workflow(EModelWorkflow("t", "p", "c", state="running"))
    stored = forge.active("EModelWorkflow")[0]
    assert forge.payload_of(stored)["state"] == "running"
    assert not forge.updated


def test_update_workflow_without_generated_resources_uses_entity_schema(ap, forge):
    resource = add_workflow(ap, forge)
    ap.store_or_update_emodel_workflow(EModelWorkflow("t", "p", "c", state="done"))
    updated = forge.active("EModelWorkflow")[0]
    assert updated.id == resource.id
    assert updated.state == "done"
    assert [part.id for part in updated.hasPart] == ["t", "p", "c"]
    assert forge.payload_of(updated)["state"] == "done"
    assert forge.schema_ids == ["schema:Entity"]


def test_update_workflow_with_generated_resources_uses_workflow_schema(ap, forge):
    add_workflow(ap, forge)
    workflow = EModelWorkflow("t", "p", "c")
    workflow.add_emodel_id("em:1")
    ap.store_or_update_emodel_workflow(workflow)
    assert forge.active("EModelWorkflow")[0].generates[0].id == "em:1"
    assert forge.schema_ids == ["schema:EModelWorkflow"]


def test_update_workflow_without_any_link(ap, forge):
    resource = add_workflow(ap, forge)
    ap.store_or_update_emodel_workflow(EModelWorkflow(None, None, None, state="done"))
    assert forge.active("EModelWorkflow")[0].id == resource.id
    assert forge.schema_ids == ["schema:Entity"]


# E-models


def test_get_emodel_by_seed(ap, forge):
    add_emodel(ap, forge, seed=1)
    add_emodel(ap, forge, seed=2)
    emodel = ap.get_emodel(seed=2)
    assert emodel.seed == 2
    assert emodel.emodel_metadata.emodel == "L5_TPC"
    assert emodel.emodel_metadata is not ap.emodel_metadata


def test_get_emodel_without_seed_needs_a_unique_match(ap, forge):
    add_emodel(ap, forge, seed=1)
    assert ap.get_emodel().seed == 1
    add_emodel(ap, forge, seed=2)
    with pytest.raises(AccessPointException, match="More than one"):
        ap.get_emodel()


def test_get_emodels(ap, forge):
    add_emodel(ap, forge, seed=1)
    add_emodel(ap, forge, seed=2)
    emodels = ap.get_emodels()
    assert [e.seed for e in emodels] == [1, 2]
    assert all(e.emodel_metadata.iteration == "v0" for e in emodels)


def test_has_best_model(ap, forge):
    add_emodel(ap, forge, seed=1)
    assert ap.has_best_model(1) is True
    assert ap.has_best_model(2) is False


@pytest.mark.parametrize("passed, expected", [(True, True), (False, True), (None, False)])
def test_is_checked_by_validation(ap, forge, passed, expected):
    add_emodel(ap, forge, seed=1, passed=passed)
    assert ap.is_checked_by_validation(1) is expected


def test_is_checked_by_validation_missing_model(ap):
    assert ap.is_checked_by_validation(1) is False


@pytest.mark.parametrize("n_model, expected", [(2, True), (3, False)])
def test_is_validated_counts_passing_models(ap, forge, n_model, expected):
    ap.pipeline_settings.n_model = n_model
    add_emodel(ap, forge, seed=1, passed=True)
    add_emodel(ap, forge, seed=2, passed=True)
    add_emodel(ap, forge, seed=3, passed=False)
    assert ap.is_validated() is expected


def test_is_validated_when_models_cannot_be_read(ap, monkeypatch):
    def broken():
        raise TypeError

    monkeypatch.setattr(ap, "get_emodels", broken)
    assert ap.is_validated() is False


def test_store_emodel_requires_workflow(ap):
    with pytest.raises(AccessPointException, match="No EModelWorkflow"):
        ap.store_emodel(make_emodel(ap))


def test_store_emodel_links_it_to_workflow(ap, forge):
    workflow = add_workflow(ap, forge)
    emodel = make_emodel(ap, seed=5)
    ap.store_emodel(emodel, description="best")
    stored = forge.active("EModel")[0]
    assert emodel.workflow_id == workflow.id
    assert (stored.seed, stored.description) == (5, "best")
    assert forge.payload_of(forge.active("EModelWorkflow")[0])["emodels"] == [stored.id]


def test_store_or_update_emodel_stores_when_absent(ap, forge):
    add_workflow(ap, forge)
    ap.store_or_update_emodel(make_emodel(ap, seed=5))
    assert forge.active("EModel")[0].seed == 5
    assert forge.active("EModel")[0].id not in forge.updated


def test_store_or_update_emodel_updates_score(ap, forge):
    resource = add_emodel(ap, forge, seed=5, fitness=2.0)
    resource.score = 2.0
    ap.store_or_update_emodel(make_emodel(ap, seed=5, fitness=9.0))
    assert forge.active("EModel")[0].score == 9.0
    assert forge.updated == [resource.id]
    assert forge.schema_ids == ["schema:EModel"]


def test_update_emodel_images_without_model(ap, forge):
    ap.update_emodel_images(seed=1)
    assert not forge.updated


@pytest.mark.parametrize("keep, expected", [(False, []), (True, "old.pdf")])
def test_update_emodel_images_replaces_old_images_unless_told(ap, forge, keep, expected):
    resource = add_emodel(ap, forge, seed=1)
    resource.image = "old.pdf"
    ap.update_emodel_images(seed=1, keep_old_images=keep)
    assert to_json(forge.active("EModel")[0]).get("image") == expected
    assert forge.updated == [resource.id]


# Scripts


def test_get_hoc(ap, forge):
    forge.add("EModelScript", payload={"seed": 2}, **identity(ap, seed=2))
    assert ap.get_hoc(seed=2).seed == 2
    assert ap.sonata_exists(2) is True
    assert ap.sonata_exists(3) is False


def test_get_hoc_without_seed(ap, forge):
    forge.add("EModelScript", payload={"seed": 2}, **identity(ap, seed=2))
    assert ap.get_hoc().seed == 2


@pytest.fixture
def hoc_paths(ap, monkeypatch, workspace):
    """Point the hoc export helpers at a local file and record the output directories used."""
    used = []

    def get_output_path(emodel, output_base_dir, use_allen_notation):
        used.append((emodel.emodel_metadata.emodel, output_base_dir))
        return workspace

    monkeypatch.setattr(nexus_module, "copy_hocs_to_new_output_path", lambda *_: None)
    monkeypatch.setattr(nexus_module, "get_output_path", get_output_path)
    monkeypatch.setattr(nexus_module, "get_hoc_file_path", lambda _: workspace / "model.hoc")
    return used


def test_store_hocs_requires_workflow(ap):
    with pytest.raises(AccessPointException, match="No EModelWorkflow"):
        ap.store_hocs()


def test_store_hocs_without_models_only_updates_workflow(ap, forge, caplog):
    add_workflow(ap, forge)
    with caplog.at_level("WARNING", logger="__main__"):
        ap.store_hocs()
    assert "No hoc file will be registered" in caplog.text
    assert not forge.active("EModelScript")


def test_store_hocs_skips_missing_file(ap, forge, hoc_paths, caplog):
    add_workflow(ap, forge)
    add_emodel(ap, forge, seed=1)
    with caplog.at_level("WARNING", logger="__main__"):
        ap.store_hocs()
    assert "Could not find the hoc file" in caplog.text
    assert not forge.active("EModelScript")


def test_store_hocs_registers_script_with_currents(ap, forge, hoc_paths, workspace):
    (workspace / "model.hoc").write_text("// hoc")
    add_workflow(ap, forge)
    add_emodel(ap, forge, seed=1, features={HOLDING_KEY: -0.1, THRESHOLD_KEY: 0.2})
    ap.store_hocs(description="exported", new_emodel_name="renamed")
    script = forge.active("EModelScript")[0]
    assert (script.seed, script.description) == (1, "exported")
    assert (script.holding_current, script.threshold_current) == (-0.1, 0.2)
    assert hoc_paths == [("renamed", "export_emodels_hoc")]
    assert forge.payload_of(forge.active("EModelWorkflow")[0])["emodel_scripts_id"] == [script.id]


def test_store_hocs_without_currents(ap, forge, hoc_paths, workspace):
    (workspace / "model.hoc").write_text("// hoc")
    add_workflow(ap, forge)
    add_emodel(ap, forge, seed=1)
    ap.store_hocs()
    assert not hasattr(forge.active("EModelScript")[0], "holding_current")


@pytest.mark.parametrize(
    "method, directory",
    [
        ("store_emodels_hoc", "export_emodels_hoc"),
        ("store_emodels_sonata", "export_emodels_sonata"),
    ],
)
def test_store_exports_use_their_own_directory(ap, monkeypatch, method, directory):
    calls = []
    monkeypatch.setattr(ap, "store_hocs", lambda *args: calls.append(args))
    getattr(ap, method)(only_best=False, description="d")
    assert calls == [(False, False, None, map, None, "d", directory)]


# Synapse model and mechanisms


def test_synapse_model_missing(ap):
    with pytest.raises(AccessPointException, match="ProbAMPANMDA_EMS not found"):
        ap.get_ProbAMPANMDA_EMS_resource()


def test_synapse_model_takes_first_of_several(ap, forge, caplog):
    first = forge.add("SynapsePhysiologyModel", name="ProbAMPANMDA_EMS")
    forge.add("SynapsePhysiologyModel", name="ProbAMPANMDA_EMS")
    with caplog.at_level("WARNING", logger="__main__"):
        assert ap.get_ProbAMPANMDA_EMS_resource() is first
    assert "Keeping the 1st resource" in caplog.text


def test_download_synapse_model_keeps_expected_file_name(ap, forge):
    forge.add(
        "SynapsePhysiologyModel", name="ProbAMPANMDA_EMS", files={"ProbAMPANMDA_EMS.mod": b"x"}
    )
    assert ap.download_ProbAMPANMDA_EMS() is True
    assert (ap.get_mechanisms_directory() / "ProbAMPANMDA_EMS.mod").read_bytes() == b"x"


def test_download_synapse_model_skips_existing_file(ap, forge):
    forge.add("SynapsePhysiologyModel", name="ProbAMPANMDA_EMS", files={"other.mod": b"x"})
    directory = ap.get_mechanisms_directory()
    directory.mkdir()
    (directory / "ProbAMPANMDA_EMS.mod").write_bytes(b"local")
    assert ap.download_ProbAMPANMDA_EMS() is False
    assert not (directory / "other.mod").exists()


def test_download_mechanisms_by_id_then_compiles(ap, forge, compile_calls):
    resource = forge.add("SubCellularModelScript", name="NaTg", files={"Na_v2.mod": b"x"})
    mechanisms = [
        MechanismConfiguration("pas", "all"),
        MechanismConfiguration("NaTg", "all", id=resource.id),
    ]
    start = os.getcwd()
    ap.download_mechanisms(mechanisms)
    assert (ap.get_mechanisms_directory() / "NaTg.mod").read_bytes() == b"x"
    assert compile_calls == [("nrnivmodl mechanisms", str(ap.get_mechanisms_directory().parent))]
    assert os.getcwd() == start


@pytest.mark.parametrize("deprecated", [True, False])
def test_download_mechanisms_looks_up_replacement(ap, forge, compile_calls, deprecated):
    replacement = add_mechanism(forge, files={"NaTg.mod": b"new"})
    id_ = None
    if deprecated:
        old = forge.add("Old", files={"NaTg.mod": b"old"})
        old._store_metadata["_deprecated"] = True
        id_ = old.id
    ap.download_mechanisms([MechanismConfiguration("NaTg", "all", id=id_)])
    assert (ap.get_mechanisms_directory() / "NaTg.mod").read_bytes() == b"new"
    assert replacement in forge.resources


def test_download_mechanisms_keeps_existing_file_and_does_not_compile(ap, forge, compile_calls):
    resource = forge.add("SubCellularModelScript", name="NaTg", files={"NaTg.mod": b"new"})
    directory = ap.get_mechanisms_directory()
    directory.mkdir()
    (directory / "NaTg.mod").write_bytes(b"local")
    ap.download_mechanisms([MechanismConfiguration("NaTg", "all", id=resource.id)])
    assert (directory / "NaTg.mod").read_bytes() == b"local"
    assert not compile_calls


def test_download_mechanisms_compiles_if_something_else_was_downloaded(ap, compile_calls):
    directory = ap.get_mechanisms_directory()
    directory.mkdir()
    ap.download_mechanisms([], any_downloaded=True)
    assert len(compile_calls) == 1


# Morphology and trace downloads


def test_download_morphology_needs_name_or_id(ap):
    with pytest.raises(TypeError, match="at least name or id_"):
        ap.download_morphology()


def species_filter():
    return {"subject": {"species": {"label": MOUSE}}}


def test_download_morphology_by_name(ap, forge):
    add_morphology(forge, **species_filter())
    assert ap.download_morphology(name="m1") == str(ap.download_directory / "m1.asc")


def test_download_morphology_ignores_other_species(ap, forge):
    add_morphology(forge, subject={"species": {"label": "Rattus norvegicus"}})
    with pytest.raises(AccessPointException, match="Could not get resource"):
        ap.download_morphology(name="m1")


def test_download_morphology_prefers_curated_duplicate(ap, forge):
    add_morphology(forge, id_="a", files={"a.asc": b""}, **species_filter())
    add_morphology(forge, id_="b", files={"b.asc": b""}, derivation="x", **species_filter())
    assert ap.download_morphology(name="m1").endswith("b.asc")


def test_download_morphology_without_curated_duplicate(ap, forge):
    add_morphology(forge, id_="a", **species_filter())
    add_morphology(forge, id_="b", **species_filter())
    with pytest.raises(AccessPointException, match="Could not get resource"):
        ap.download_morphology(name="m1")


def test_download_morphology_selects_requested_format(ap, forge):
    add_morphology(forge, files={"m1.h5": b"", "m1.swc": b""})
    assert ap.download_morphology(id_="morph:1", format_="swc").endswith("m1.swc")


def test_download_morphology_accepts_upper_case_suffix(ap, forge):
    add_morphology(forge, files={"m1.h5": b"", "m1.ASC": b""})
    # Case-insensitive filesystems (macOS) report the lower-case suffix as existing.
    assert ap.download_morphology(id_="morph:1", format_="asc").lower().endswith("m1.asc")


def test_download_morphology_avoids_h5_by_default(ap, forge):
    add_morphology(forge, files={"m1.h5": b"", "m1.asc": b""})
    assert ap.download_morphology(id_="morph:1").endswith("m1.asc")


def test_download_morphology_with_only_h5(ap, forge):
    add_morphology(forge, files={"m1.h5": b""})
    with pytest.raises(FileNotFoundError, match="Could not find morphology"):
        ap.download_morphology(id_="morph:1")


def test_download_trace_by_id(ap, nexus):
    trace = nwb_trace(ap, nexus)
    assert ap.download_trace(id_=trace.id).endswith("c1.nwb")


def test_download_trace_by_legacy_id(ap, nexus):
    trace = nwb_trace(ap, nexus)
    assert ap.download_trace(id_legacy=trace.id).endswith("c1.nwb")


def test_download_trace_by_name_in_later_project(ap, nexus):
    nexus.forges["public/thalamus"].add("Trace", name="c9", files={"c9.nwb": b""})
    assert ap.download_trace(name="c9").endswith("c9.nwb")


def test_download_trace_needs_a_key(ap):
    with pytest.raises(TypeError, match="At least id_ or name"):
        ap.download_trace()


def test_download_trace_not_found(ap):
    with pytest.raises(ValueError, match="No matching resource"):
        ap.download_trace(id_="unknown")


# Gene expression and channel maps


def test_load_channel_gene_expression(ap, forge):
    content = b"me-type,t-type,modality,value\nL5,t1,rna,3\n"
    forge.add("RNASequencing", name="table", files={"table.csv": content})
    df, path = ap.load_channel_gene_expression("table")
    assert df.loc[("L5", "t1", "rna"), "value"] == 3
    assert path == str(ap.download_directory / "table.csv")


def test_load_ic_map(ap, forge):
    forge.add("IonChannelMapping", name="icmapping", files={"ic.json": b"{}"})
    assert ap.load_ic_map() == str(ap.download_directory / "ic.json")


# Catalogues


def test_get_available_morphologies(ap, forge, caplog):
    with caplog.at_level("WARNING", logger="__main__"):
        assert ap.get_available_morphologies() == set()
    assert "Did not find any" in caplog.text
    add_morphology(forge)
    assert ap.get_available_morphologies() == {"m1"}


def test_get_available_mechanisms_none(ap):
    assert ap.get_available_mechanisms() is None


def test_get_available_mechanisms_reads_exposed_parameters(ap, forge):
    resource = forge.add(
        "SubCellularModelScript",
        name="NaTg",
        modelId="v1",
        temperature={"value": 34},
        isLjpCorrected=True,
        stochastic=False,
        exposesParameter=[
            {"type": "ConductanceDensity", "name": "gbar", "lowerLimit": 0, "upperLimit": 1},
            {"type": "ConductanceDensity", "name": "g"},
            {"type": "CurrentDensity", "name": "ina", "ionName": "na"},
            {"type": "CurrentDensity", "name": "ihcn", "ionName": "non-specific"},
            {"type": "CurrentDensity", "name": "unknown"},
            {"type": "IonConcentration", "name": "cai"},
            {"type": "Other", "name": "ignored"},
        ],
    )
    mechanism = ap.get_available_mechanisms()[0]
    assert mechanism.id == resource.id
    assert (mechanism.version, mechanism.temperature, mechanism.ljp_corrected) == ("v1", 34, True)
    assert mechanism.stochastic is False
    assert mechanism.parameters == {"gbar_NaTg": [0, 1], "g_NaTg": [None, None]}
    assert mechanism.ion_currents == ["ina"]
    assert sorted(mechanism.ionic_concentrations) == ["cai", "nai"]


def test_get_available_mechanisms_handles_sparse_resources(ap, forge):
    forge.add(
        "SubCellularModelScript",
        name="Ih",
        temperature=34,
        exposesParameter={"type": "ConductanceDensity", "name": "gbar"},
    )
    forge.add("SubCellularModelScript", name="Bare")
    mechanisms = {m.name: m for m in ap.get_available_mechanisms()}
    assert mechanisms["Ih"].temperature == 34
    assert mechanisms["Ih"].parameters == {"gbar_Ih": [None, None]}
    assert mechanisms["Bare"].version is None
    assert mechanisms["Bare"].parameters == {}


def test_get_available_mechanisms_stays_in_project_and_applies_filters(ap, forge):
    forge.add("SubCellularModelScript", name="Ih", kind="a")
    forge.add("SubCellularModelScript", name="Na", kind="b")
    assert [m.name for m in ap.get_available_mechanisms({"kind": "b"})] == ["Na"]
    assert forge.search_calls[-1]["cross_bucket"] is False


def test_get_available_traces_reads_metadata(ap, nexus):
    nwb_trace(
        ap,
        nexus,
        stimulus=[{"stimulusType": {"label": "IDRest"}}, {"stimulusType": {"id": "x"}}],
        brainLocation={"brainRegion": {"label": "SS"}},
        annotation=[
            {"name": "M-type annotation", "hasBody": {"label": "L5"}},
            {"name": "E-Type annotation", "hasBody": {"label": "cAC"}},
        ],
    )
    trace = ap.get_available_traces()[0]
    assert trace.cell_name == "c1"
    assert trace.ecodes == {"IDRest": {}}
    assert trace.etype == "cAC"
    assert trace.species.label == MOUSE
    assert trace.brain_region.brainRegion.label == "SS"


def test_get_available_traces_handles_sparse_resources(ap, nexus):
    nexus.traces.add(
        "ExperimentalTrace",
        name="c1",
        files={"c1.nwb": b""},
        stimulus={"stimulusType": {"label": "IDRest"}},
        annotation={"name": "M-type annotation", "hasBody": {"label": "L5"}},
    )
    trace = ap.get_available_traces(filter_species=False)[0]
    assert trace.ecodes == {"IDRest": {}}
    assert trace.etype is None
    assert trace.species is None
    assert trace.brain_region is None


def test_get_available_traces_applies_brain_region(ap, nexus):
    nwb_trace(ap, nexus, brainLocation=ap.emodel_metadata_ontology.brain_region)
    nwb_trace(ap, nexus, name="c2", brainLocation={"type": "BrainLocation"})
    names = [t.cell_name for t in ap.get_available_traces(filter_brain_region=True)]
    assert names == ["c1"]


def test_get_available_traces_when_lookup_returns_nothing(ap, monkeypatch):
    monkeypatch.setattr(nexus_module, "get_available_traces", lambda **_: None)
    assert ap.get_available_traces() == []


# Morphology registration


@pytest.mark.parametrize(
    "reconstructed, kind",
    [(True, "ReconstructedNeuronMorphology"), (False, "PlaceholderNeuronMorphology")],
)
def test_store_morphology(ap, forge, workspace, reconstructed, kind):
    path = workspace / "cell.asc"
    path.write_text("x")
    ap.store_morphology("cell", str(path), reconstructed=reconstructed)
    stored = forge.active(kind)[0]
    assert stored.name == "cell"
    assert "NeuronMorphology" in stored.type
    assert stored.distribution.name == "cell.asc"
    assert not hasattr(stored, "annotation")
    assert forge.schema_ids == ["schema:NeuronMorphology"]


def test_store_morphology_annotates_mtype(ap, forge, workspace):
    path = workspace / "cell.asc"
    path.write_text("x")
    ap.store_morphology("cell", str(path), mtype="L5_TPC")
    annotation = to_json(forge.active("NeuronMorphology")[0])["annotation"]
    assert annotation[0]["hasBody"]["label"] == "L5_TPC"


# Regressions


REQUIRED_CONFIGURATIONS = (
    "FitnessCalculatorConfiguration",
    "EModelConfiguration",
    "EModelPipelineSettings",
    "ExtractionTargetsConfiguration",
)


def quality_labels(resource):
    annotations = to_json(resource).get("annotation", [])
    return [a["hasBody"]["label"] for a in annotations if "QualityAnnotation" in str(a)]


def test_complete_emodel_is_tagged_analysis_suitable_and_curated(ap, forge):
    add_workflow(ap, forge)
    for type_ in REQUIRED_CONFIGURATIONS:
        forge.add(type_, **identity(ap))
    ap.store_emodel(make_emodel(ap))
    assert quality_labels(forge.active("EModel")[0]) == ["Analysis Suitable", "Curated"]


@pytest.mark.parametrize("missing", REQUIRED_CONFIGURATIONS)
def test_incomplete_emodel_is_not_tagged(ap, forge, missing):
    add_workflow(ap, forge)
    for type_ in REQUIRED_CONFIGURATIONS:
        if type_ != missing:
            forge.add(type_, **identity(ap))
    ap.store_emodel(make_emodel(ap))
    assert quality_labels(forge.active("EModel")[0]) == []


def test_store_emodel_finds_configurations_stored_with_legacy_metadata(ap, forge):
    add_workflow(ap, forge)
    legacy = ap.emodel_metadata_ontology.filters_for_resource_legacy()
    for type_ in REQUIRED_CONFIGURATIONS:
        forge.add(type_, **legacy)
    ap.store_emodel(make_emodel(ap))
    assert quality_labels(forge.active("EModel")[0]) == ["Analysis Suitable", "Curated"]


def test_store_or_update_emodel_replaces_the_stored_json(ap, forge):
    add_emodel(ap, forge, seed=5, fitness=2.0)
    ap.store_or_update_emodel(make_emodel(ap, seed=5, fitness=9.0))
    stored = forge.active("EModel")[0]
    assert len(stored.distribution) == 1
    assert forge.payload_of(stored)["fitness"] == 9.0


@pytest.fixture
def emodel_figures(monkeypatch):
    monkeypatch.setattr(EModel, "build_pdf_dependencies", lambda self, seed: ["fig__traces.pdf"])


def test_store_or_update_emodel_attaches_images(ap, forge, emodel_figures):
    add_emodel(ap, forge, seed=5)
    ap.store_or_update_emodel(make_emodel(ap, seed=5, fitness=9.0))
    assert forge.active("EModel")[0].image.about == "traces"


def test_update_emodel_images_replaces_images(ap, forge, emodel_figures):
    resource = add_emodel(ap, forge, seed=1)
    resource.image = "old.pdf"
    ap.update_emodel_images(seed=1)
    images = forge.active("EModel")[0].image
    assert [image.about for image in images] == ["traces"]


def test_update_emodel_images_can_keep_old_images(ap, forge, emodel_figures):
    resource = add_emodel(ap, forge, seed=1)
    resource.image = "old.pdf"
    ap.update_emodel_images(seed=1, keep_old_images=True)
    assert to_json(forge.active("EModel")[0])["image"][0] == "old.pdf"
    assert forge.active("EModel")[0].image[1].about == "traces"


def test_get_t_types_matches_normalised_me_type(ap, forge):
    content = (
        b"me-type,t-type,modality,value\n"
        b"L5_TPC,t1,rna,3\n"
        b"L5-TPC,t2,rna,1\n"
        b"L5_TPC,t1,atac,2\n"
        b"other,t3,rna,1\n"
    )
    forge.add("RNASequencing", name="table", files={"table.csv": content})
    assert ap.get_t_types("table") == ["t1", "t2"]


def test_get_available_mechanisms_keeps_nonspecific_currents(ap, forge):
    forge.add(
        "SubCellularModelScript",
        name="Ih",
        exposesParameter={"type": "CurrentDensity", "name": "ihcn", "ionName": "non-specific"},
    )
    assert ap.get_available_mechanisms()[0].nonspecific_currents == ["ihcn"]


def test_missing_trace_error_names_the_etype(ap, forge):
    payload = targets_payload([])
    forge.add("ExtractionTargetsConfiguration", payload=payload, **identity(ap))
    with pytest.raises(AccessPointException, match="with etype cAC"):
        ap.get_targets_configuration()
