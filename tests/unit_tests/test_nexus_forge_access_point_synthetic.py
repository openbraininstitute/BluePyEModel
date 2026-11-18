"""Tests of ``forge_access_point`` against an in-memory Nexus. No network or credentials."""

import json
import pathlib

import pytest
from kgforge.core import Resource
from kgforge.core.commons.strategies import ResolvingStrategy
from kgforge.specializations.resources import Dataset

from bluepyemodel.access_point import forge_access_point as fap
from bluepyemodel.access_point.forge_access_point import AccessPointException
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint
from bluepyemodel.emodel_pipeline.emodel import EModel
from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata
from bluepyemodel.emodel_pipeline.emodel_workflow import EModelWorkflow
from tests.unit_tests.nexus_fakes import FakeNexus
from tests.unit_tests.nexus_fakes import make_token

ATLAS_ID = "https://bbp.epfl.ch/neurosciencegraph/data/4906ab85-694f-469d-962f-c0174e901885"
METADATA_STR = "emodel=L5__iteration=v0__"


@pytest.fixture
def nexus(monkeypatch):
    world = FakeNexus()
    monkeypatch.setattr(fap, "KnowledgeGraphForge", world.connect)
    monkeypatch.setattr(NexusForgeAccessPoint, "forges", {})
    return world


@pytest.fixture
def ap(nexus):
    return NexusForgeAccessPoint(project="emodel", organisation="demo", access_token=make_token())


@pytest.fixture
def forge(nexus):
    return nexus.project


def make_emodel(seed=3):
    return EModel(
        fitness=4.0,
        parameter={"g": 1.0},
        score={"a": 1.5, "b": 2.5},
        features={"f": 0.1},
        scoreValidation={},
        passedValidation=True,
        seed=seed,
        emodel_metadata=EModelMetadata(emodel="L5", iteration_tag="v0"),
    )


# Connection and token handling


def test_forge_connection_is_cached_while_token_is_valid(nexus, ap):
    connections = len(nexus.connections)
    assert ap.forge is ap.forge
    assert len(nexus.connections) == connections == 1


def test_forge_reconnects_when_token_is_close_to_expiry(nexus):
    # The cache keeps a forge only until 15 minutes before the token expires.
    ap = NexusForgeAccessPoint(project="p", organisation="o", access_token=make_token(600))
    connections = len(nexus.connections)
    _ = ap.forge
    assert len(nexus.connections) == connections + 1


def test_expired_token_is_replaced_before_connecting(nexus, monkeypatch):
    fresh = make_token()
    monkeypatch.setattr(NexusForgeAccessPoint, "get_access_token", staticmethod(lambda: fresh))
    ap = NexusForgeAccessPoint(project="p", organisation="o", access_token=make_token(-10))
    assert ap.access_token == fresh
    assert nexus.connections[-1].token == fresh


def test_token_is_requested_when_none_is_given(nexus, monkeypatch):
    token = make_token()
    monkeypatch.setattr(NexusForgeAccessPoint, "get_access_token", staticmethod(lambda: token))
    ap = NexusForgeAccessPoint(project="p", organisation="o")
    assert ap.access_token == token


def test_agent_keeps_only_identity_fields(ap):
    assert ap.agent.preferred_username == "test_user"
    assert ap.agent.type == ["Person", "Agent"]
    assert ap.agent.id.endswith("/users/test_user")
    assert not hasattr(ap.agent, "exp")


def test_get_access_token_prefers_stored_token(monkeypatch):
    monkeypatch.setattr(fap, "refresh_token", lambda: "stored")
    assert NexusForgeAccessPoint.get_access_token() == "stored"


@pytest.mark.parametrize("failing", [True, False])
def test_get_access_token_falls_back_to_prompt(monkeypatch, failing):
    def no_token():
        if failing:
            raise RuntimeError("no workflow")

    monkeypatch.setattr(fap, "refresh_token", no_token)
    monkeypatch.setattr(fap.getpass, "getpass", lambda: "typed")
    assert NexusForgeAccessPoint.get_access_token() == "typed"


def test_connect_forge_uses_default_configuration(nexus):
    NexusForgeAccessPoint.connect_forge("o/p", "https://e", "token")
    assert nexus.connections[-1].forge_path.endswith("prod-forge-nexus.yml")
    NexusForgeAccessPoint.connect_forge("o/p", "https://e", "token", forge_path="mine.yml")
    assert nexus.connections[-1].forge_path == "mine.yml"


# Cached ontology lookups


@pytest.mark.parametrize(
    "attribute, marker, field",
    [
        ("available_etypes", "EType", "e_type_id"),
        ("available_mtypes", "MType", "m_type_id"),
        ("available_ttypes", "BrainCellTranscriptomeType", "t_type_id"),
    ],
)
def test_available_types_are_queried_once(ap, forge, attribute, marker, field):
    forge.sparql_answers = [((marker,), [Resource(**{field: "id1"}), Resource(**{field: "id2"})])]
    assert getattr(ap, attribute) == ["id1", "id2"]
    assert getattr(ap, attribute) == ["id1", "id2"]
    assert len(forge.sparql_calls) == 1


@pytest.mark.parametrize(
    "method", ["get_available_etypes", "get_available_mtypes", "get_available_ttypes"]
)
def test_available_types_empty_when_query_fails(ap, method):
    assert getattr(ap, method)() == []


def test_atlas_release_records_revision_once(ap, nexus):
    nexus.atlas.add("AtlasRelease", id_=ATLAS_ID)
    assert ap.atlas_release["_rev"] == 1
    assert ap.atlas_release["id"] == ATLAS_ID
    assert nexus.atlas.retrieve_calls == [ATLAS_ID]


# Resource construction


def test_add_contribution_on_resource(ap):
    resource = ap.add_contribution(Resource(type="X"))
    assert resource.contribution.type == "Contribution"
    assert resource.contribution.agent is ap.agent


def test_add_contribution_on_dataset(ap, forge):
    dataset = ap.add_contribution(Dataset(forge, type="X"))
    assert dataset.contribution.agent.id == ap.agent.id


def test_add_contribution_without_agent(ap):
    ap.agent = None
    assert not hasattr(ap.add_contribution(Resource(type="X")), "contribution")


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("all", ResolvingStrategy.ALL_MATCHES),
        ("best", ResolvingStrategy.BEST_MATCH),
        ("exact", ResolvingStrategy.EXACT_MATCH),
    ],
)
def test_resolve_maps_strategy(ap, forge, strategy, expected):
    ap.resolve("cAC", strategy=strategy)
    assert forge.resolve_calls[-1]["strategy"] is expected


def test_resolve_rejects_unknown_strategy(ap):
    with pytest.raises(ValueError, match="does not exist"):
        ap.resolve("cAC", strategy="fuzzy")


def test_add_images_uses_suffix_of_file_name(ap, forge):
    dataset = Dataset(forge, type="EModel")
    ap.add_images_to_resource(["/a/plot__traces.pdf", "/a/plot__scores.png"], dataset)
    stored = forge.register(dataset)
    assert [(i.about, i.encodingFormat) for i in stored.image] == [
        ("traces", "application/pdf"),
        ("scores", "application/png"),
    ]


# register


def test_register_requires_type(ap):
    with pytest.raises(AccessPointException, match="should contain 'type'"):
        ap.register({"name": "x"})


def test_register_stores_resource_with_common_fields(ap, forge):
    ap.register({"type": ["Entity", "Thing"], "name": "x"}, type_="Thing")
    stored = forge.active("Thing")[0]
    assert stored.name == "x"
    assert stored.objectOfStudy.label == "Single Cell"
    assert stored.contribution.type == "Contribution"
    assert forge.schema_ids == ["schema:Thing"]


def test_register_adds_atlas_release_only_with_brain_location(ap, forge, nexus):
    nexus.atlas.add("AtlasRelease", id_=ATLAS_ID)
    ap.register({"type": "Plain", "name": "x"})
    ap.register({"type": "Located", "name": "y", "brainLocation": {"label": "SS"}})
    assert not hasattr(forge.active("Plain")[0], "atlasRelease")
    assert forge.active("Located")[0].atlasRelease.id == ATLAS_ID


def test_register_ignores_existing_resource_without_replace(ap, forge, caplog):
    existing = forge.add("Thing", name="x")
    with caplog.at_level("WARNING", logger="__main__"):
        ap.register({"type": "Thing", "name": "x"}, filters_existence={"type": "Thing"})
    assert forge.active("Thing") == [existing]
    assert "already exist" in caplog.text


def test_register_replaces_existing_resource(ap, forge):
    existing = forge.add("Thing", name="x")
    ap.register({"type": "Thing", "name": "new"}, filters_existence={"type": "Thing"}, replace=True)
    assert forge.deprecated == [existing.id]
    assert [r.name for r in forge.active("Thing")] == ["new"]


def test_register_finds_existing_resource_through_legacy_filters(ap, forge):
    forge.add("Thing", emodel="old")
    ap.register(
        {"type": "Thing", "name": "new"},
        filters_existence={"type": "Thing", "eModel": "old"},
        legacy_filters_existence={"type": "Thing", "emodel": "old"},
    )
    assert len(forge.active("Thing")) == 1


def test_register_uploads_distributions_and_images(ap, forge, workspace):
    data = workspace / "data.json"
    data.write_text('{"a": 1}')
    ap.register(
        {"type": "Thing", "name": "x"},
        distributions=[str(data)],
        images=[str(workspace / "fig__traces.pdf")],
    )
    stored = forge.active("Thing")[0]
    assert stored.distribution.encodingFormat == "application/json"
    assert forge.payload_of(stored) == {"a": 1}
    assert stored.image.about == "traces"


def test_register_validates_workflow_against_entity_schema(ap, forge):
    ap.register({"type": "EModelWorkflow", "name": "x"}, type_="EModelWorkflow")
    assert forge.schema_ids == ["schema:Entity"]


# Fetching


def test_retrieve(ap, forge):
    resource = forge.add("Thing")
    assert ap.retrieve(resource.id) is resource
    assert ap.retrieve("unknown") is None


def test_fetch_requires_type_or_id(ap):
    with pytest.raises(AccessPointException, match="either 'type' or 'id'"):
        ap.fetch({"name": "x"})


def test_fetch_passes_search_options(ap, forge):
    forge.add("Thing")
    assert len(ap.fetch({"type": "Thing"})) == 1
    assert forge.search_calls[-1]["cross_bucket"] is True
    assert forge.search_calls[-1]["limit"] == 5000
    assert forge.search_calls[-1]["search_endpoint"] == "sparql"
    ap.fetch({"type": "Thing"}, cross_bucket=False)
    assert forge.search_calls[-1]["cross_bucket"] is False


def test_fetch_returns_none_when_nothing_matches(ap):
    assert ap.fetch({"type": "Thing"}) is None


def test_fetch_legacy_compatible_only_falls_back_when_needed(ap, forge):
    new = forge.add("New")
    legacy = forge.add("Legacy")
    assert ap.fetch_legacy_compatible({"type": "New"}, {"type": "Legacy"}) == [new]
    assert ap.fetch_legacy_compatible({"type": "Missing"}, {"type": "Legacy"}) == [legacy]
    assert ap.fetch_legacy_compatible({"type": "Missing"}) is None


def test_fetch_one_strict_errors(ap, forge):
    forge.add("Thing")
    forge.add("Thing")
    with pytest.raises(AccessPointException, match="Could not get"):
        ap.fetch_one({"type": "Missing"})
    with pytest.raises(AccessPointException, match="More than one"):
        ap.fetch_one({"type": "Thing"})


def test_fetch_one_non_strict(ap, forge):
    first = forge.add("Thing")
    forge.add("Thing")
    assert ap.fetch_one({"type": "Missing"}, strict=False) is None
    assert ap.fetch_one({"type": "Thing"}, strict=False) is first


def test_fetch_one_single_resource(ap, forge):
    resource = forge.add("Thing")
    assert ap.fetch_one({"type": "Thing"}) is resource


# Downloading


def test_download_unknown_resource(ap, tmp_path):
    with pytest.raises(AccessPointException, match="Could not find resource"):
        ap.download("unknown", tmp_path)


def test_download_without_distribution(ap, forge, tmp_path):
    assert ap.download(forge.add("Thing").id, tmp_path) == []


def test_download_single_distribution(ap, forge, tmp_path):
    resource = forge.add("Thing", files={"a.txt": b"hello"})
    paths = ap.download(resource.id, tmp_path)
    assert paths == [str(tmp_path / "a.txt")]
    assert pathlib.Path(paths[0]).read_bytes() == b"hello"


def test_download_several_distributions(ap, forge, tmp_path):
    resource = forge.add("Thing", files={"a.txt": b"1", "b.txt": b"2"})
    assert ap.download(resource.id, tmp_path) == [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]


def test_download_rejects_distribution_without_name(ap, forge, tmp_path):
    resource = forge.add("Thing", distribution=[{"contentUrl": "blob://x"}])
    with pytest.raises(AttributeError, match="does not have a file name"):
        ap.download(resource.id, tmp_path)


def test_download_detects_missing_file(ap, forge, tmp_path):
    resource = forge.add("Thing", distribution={"name": "a.txt", "contentUrl": "blob://gone"})
    with pytest.raises(AccessPointException, match="Download failed"):
        ap.download(resource.id, tmp_path)


def test_resource_location_requires_distribution(ap, forge, tmp_path):
    with pytest.raises(AccessPointException, match="does not have distribution"):
        ap.resource_location(forge.add("Thing"), tmp_path)


def test_resource_location_prefers_file_on_disk(ap, forge, tmp_path):
    resource = forge.add(
        "Thing",
        distribution=[
            {"name": "a.nwb", "atLocation": {"location": "file:///gpfs/a.nwb"}},
            {"name": "b.nwb", "atLocation": {"location": "/gpfs/b.nwb"}},
        ],
    )
    assert ap.resource_location(resource, tmp_path) == ["//gpfs/a.nwb", "/gpfs/b.nwb"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("at_location", [None, {"store": "s3"}])
def test_resource_location_downloads_otherwise(ap, forge, tmp_path, at_location):
    resource = forge.add("Thing", files={"a.nwb": b"1"})
    if at_location:
        resource.distribution.atLocation = Resource.from_json(at_location)
    assert ap.resource_location(resource, tmp_path) == [str(tmp_path / "a.nwb")]


# Deprecation


def test_deprecate_only_searches_own_project_and_restores_setting(ap, forge):
    resource = forge.add("Thing")
    ap.deprecate({"type": "Thing"})
    assert forge.search_calls[-1]["cross_bucket"] is False
    assert ap.cross_bucket is True
    assert forge.deprecated == [resource.id]


def test_deprecate_skips_unretrievable_resources(ap, forge, monkeypatch):
    forge.add("Thing")
    monkeypatch.setattr(ap, "retrieve", lambda id_: None)
    ap.deprecate({"type": "Thing"})
    assert forge.deprecated == []


def test_deprecate_nothing_found(ap, forge):
    ap.deprecate({"type": "Thing"})
    assert forge.deprecated == []


def test_deprecate_all_needs_confirmation(ap, forge, monkeypatch):
    resource = forge.add("EModel", eModel="L5")
    monkeypatch.setattr(fap, "yesno", lambda _: False)
    ap.deprecate_all({"eModel": "L5"})
    assert forge.deprecated == []
    monkeypatch.setattr(fap, "yesno", lambda _: True)
    ap.deprecate_all({"eModel": "L5"})
    assert forge.deprecated == [resource.id]


def test_deprecate_all_uses_legacy_metadata(ap, forge, monkeypatch):
    resource = forge.add("EModel", emodel="L5")
    monkeypatch.setattr(fap, "yesno", lambda _: True)
    ap.deprecate_all({"eModel": "L5"}, {"emodel": "L5"})
    assert forge.deprecated == [resource.id]


# Serialization


@pytest.mark.parametrize(
    "metadata, seed, expected",
    [
        ({"iteration": "v0", "eModel": "L5", "tType": "t"}, 3, "EM__v0__L5__t__3"),
        ({"iteration": "v0", "emodel": "L5", "ttype": "t"}, None, "EM__v0__L5__t"),
        ({}, 0, "EM__0"),
    ],
)
def test_resource_name(metadata, seed, expected):
    assert NexusForgeAccessPoint.resource_name("EModel", metadata, seed=seed) == expected


def test_dump_json_strips_nexus_only_fields(workspace):
    (workspace / "nexus_temp" / "v0").mkdir(parents=True)

    class Payload:
        @staticmethod
        def as_dict():
            return {"a": 1, "nexus_images": ["i.pdf"], "nexus_distributions": ["x.hoc"]}

    paths = NexusForgeAccessPoint.dump_json_and_get_distributions(
        Payload(), "EModel", METADATA_STR, seed=3
    )
    assert pathlib.Path(paths[0]).name == f"EM__{METADATA_STR}__3.json"
    assert paths[1:] == ["x.hoc"]
    assert json.loads(pathlib.Path(paths[0]).read_text()) == {"a": 1}


def test_get_seed_from_object():
    obj = make_emodel(seed=7)
    assert NexusForgeAccessPoint.get_seed_from_object(obj, "EModel") == 7
    assert NexusForgeAccessPoint.get_seed_from_object(obj, "EModelScript") == 7
    assert NexusForgeAccessPoint.get_seed_from_object(obj, "EModelWorkflow") is None


def test_object_to_nexus_round_trip(ap, forge, workspace):
    (workspace / "nexus_temp" / "v0").mkdir(parents=True)
    ap.object_to_nexus(
        make_emodel(),
        {"iteration": "v0", "eModel": "L5"},
        METADATA_STR,
        {"iteration": "v0", "emodel": "L5"},
        currents={"holding": -0.1, "threshold": 0.2},
    )
    stored = forge.active("EModel")[0]
    assert stored.name == "EM__v0__L5__3"
    assert stored.score == 4.0
    assert (stored.holding_current, stored.threshold_current) == (-0.1, 0.2)
    assert forge.schema_ids == ["schema:EModel"]

    loaded = ap.resource_to_object("EModel", stored, {}, workspace / "download")
    assert loaded.seed == 3
    assert loaded.scores == {"a": 1.5, "b": 2.5}


def test_object_to_nexus_replaces_regardless_of_annotation(ap, forge, workspace):
    (workspace / "nexus_temp" / "v0").mkdir(parents=True)
    workflow = EModelWorkflow("t", "p", "c")
    metadata = {"iteration": "v0", "eModel": "L5"}
    legacy = {"iteration": "v0", "emodel": "L5"}
    ap.object_to_nexus(workflow, {**metadata, "annotation": [{"name": "a"}]}, METADATA_STR, legacy)
    ap.object_to_nexus(workflow, {**metadata, "annotation": [{"name": "b"}]}, METADATA_STR, legacy)
    stored = forge.active("EModelWorkflow")
    assert len(stored) == 1
    assert stored[0].annotation[0].name == "b"
    assert len(forge.deprecated) == 1


def test_object_to_nexus_links_related_ids(ap, forge, workspace):
    (workspace / "nexus_temp" / "v0").mkdir(parents=True)
    ap.object_to_nexus(
        EModelWorkflow("t", None, None), {"iteration": "v0"}, METADATA_STR, {"iteration": "v0"}
    )
    assert forge.active("EModelWorkflow")[0].hasPart[0].id == "t"


def test_update_distribution_replaces_the_json(ap, forge, workspace):
    (workspace / "nexus_temp" / "v0").mkdir(parents=True)
    resource = forge.add("EModelWorkflow", payload={"state": "old"})
    updated = ap.update_distribution(
        resource, METADATA_STR, EModelWorkflow("t", "p", "c", state="new")
    )
    assert updated._store_metadata["_rev"] == 1
    assert len(updated.distribution) == 1
    assert forge.payload_of(forge.update(updated))["state"] == "new"


def test_update_distribution_accepts_single_element_list(ap, forge, workspace):
    (workspace / "nexus_temp" / "v0").mkdir(parents=True)
    resource = forge.add("EModelWorkflow", distribution=[{"name": "a.json"}])
    updated = ap.update_distribution(resource, METADATA_STR, EModelWorkflow("t", "p", "c"))
    assert len(updated.distribution) == 1


@pytest.mark.parametrize(
    "distribution, message",
    [
        ([{"name": "a.json"}, {"name": "b.json"}], "more than 1 distribution"),
        ("not-a-distribution", "requires a dict or a single-element list"),
    ],
)
def test_update_distribution_rejects_other_shapes(ap, forge, workspace, distribution, message):
    (workspace / "nexus_temp" / "v0").mkdir(parents=True)
    resource = forge.add("EModelWorkflow", distribution=distribution)
    with pytest.raises(TypeError, match=message):
        ap.update_distribution(resource, METADATA_STR, EModelWorkflow("t", "p", "c"))


def test_resource_to_object_reads_legacy_payload_from_resource(ap, forge, tmp_path):
    resource = forge.add(
        "EModelWorkflow",
        files={"unrelated.txt": b""},
        eModel="L5",
        targets_configuration_id="t",
        pipeline_settings_id="p",
        emodel_configuration_id="c",
    )
    workflow = ap.resource_to_object("EModelWorkflow", resource, {"eModel": "L5"}, tmp_path)
    assert workflow.targets_configuration_id == "t"


def test_nexus_to_object_with_legacy_metadata(ap, forge, tmp_path):
    forge.add(
        "EModelWorkflow",
        emodel="L5",
        payload={
            "targets_configuration_id": "t",
            "pipeline_settings_id": "p",
            "emodel_configuration_id": "c",
        },
    )
    workflow = ap.nexus_to_object("EModelWorkflow", {"eModel": "L5"}, tmp_path, {"emodel": "L5"})
    assert workflow.emodel_configuration_id == "c"


def test_nexus_to_object_without_legacy_metadata(ap, tmp_path):
    with pytest.raises(AccessPointException):
        ap.nexus_to_object("EModelWorkflow", {"eModel": "L5"}, tmp_path)


def test_nexus_to_objects(ap, forge, tmp_path):
    payload = {
        "targets_configuration_id": "t",
        "pipeline_settings_id": "p",
        "emodel_configuration_id": "c",
    }
    first = forge.add("EModelWorkflow", emodel="L5", payload=payload)
    second = forge.add("EModelWorkflow", emodel="L5", payload=payload)
    objects, ids = ap.nexus_to_objects(
        "EModelWorkflow", {"eModel": "L5"}, tmp_path, {"emodel": "L5"}
    )
    assert ids == [first.id, second.id]
    assert all(isinstance(o, EModelWorkflow) for o in objects)
    assert ap.nexus_to_objects("EModelWorkflow", {"eModel": "x"}, tmp_path) == ([], [])


def test_get_nexus_id(ap, forge):
    resource = forge.add("Thing", emodel="L5")
    assert ap.get_nexus_id("Thing", {"eModel": "L5"}, {"emodel": "L5"}) == resource.id
    assert ap.get_nexus_id("Thing", {}) == resource.id


# Filtering


def test_brain_region_filter():
    region = Resource(subClassOf="nsg:BrainRegion")
    resources = [region, Resource(subClassOf="nsg:Other"), Resource()]
    assert NexusForgeAccessPoint.brain_region_filter(resources) == [region]


@pytest.mark.parametrize("name", ["etype", "mtype", "ttype"])
def test_filter_resources_by_type(ap, name):
    setattr(ap, f"_available_{name}s", ["keep"])
    kept = Resource(id="keep")
    assert ap.filter_resources([kept, Resource(id="drop")], name) == [kept]


def test_filter_resources_by_brain_region(ap):
    region = Resource(subClassOf="nsg:BrainRegion")
    assert ap.filter_resources([region], "brain_region") == [region]


def test_filter_resources_rejects_unknown_filter(ap):
    with pytest.raises(AccessPointException, match="Filter not expected"):
        ap.filter_resources([], "species")


def test_type_filter_rejects_unknown_filter(ap):
    with pytest.raises(AccessPointException, match="should be in"):
        ap.type_filter([], "species")


# Module-level helpers


def test_project_specific_access_points(nexus):
    token = make_token()
    assert fap.ontology_forge_access_point(token).bucket == "neurosciencegraph/datamodels"
    assert fap.atlas_forge_access_point(token).bucket == "bbp/atlas"


def test_not_found_without_suggestions(ap):
    with pytest.raises(AccessPointException, match=r"^No match\.$"):
        fap.raise_not_found_exception("No match", "x", ap, "etype")


def test_not_found_keeps_final_period(ap):
    with pytest.raises(AccessPointException, match=r"^No match\.$"):
        fap.raise_not_found_exception("No match.", "x", ap, "etype")


@pytest.mark.parametrize("as_list", [True, False])
def test_not_found_lists_suggestions_of_right_kind(ap, forge, as_list):
    ap._available_etypes = ["good"]
    match = Resource(id="good", label="cAC")
    forge.suggestions["c"] = [match, Resource(id="bad", label="wrong")] if as_list else match
    with pytest.raises(AccessPointException, match="Maybe you meant") as error:
        fap.raise_not_found_exception("No match", "c", ap, "etype")
    assert "cAC" in str(error.value)
    assert "wrong" not in str(error.value)


def test_not_found_with_only_irrelevant_suggestions(ap, forge):
    ap._available_etypes = []
    forge.suggestions["c"] = [Resource(id="bad", label="wrong")]
    with pytest.raises(AccessPointException, match=r"^No match\.$"):
        fap.raise_not_found_exception("No match", "c", ap, "etype")


def test_check_resource_rejects_unknown_category(ap):
    with pytest.raises(AccessPointException, match="should be in"):
        fap.check_resource("cAC", "species", access_point=ap)


@pytest.mark.parametrize("category", ["etype", "mtype", "ttype"])
def test_check_resource_accepts_known_resource(ap, forge, category):
    setattr(ap, f"_available_{category}s", ["good"])
    forge.ontology["x"] = Resource(id="good")
    fap.check_resource("x", category, access_point=ap)


def test_check_resource_reports_missing_resource(ap):
    with pytest.raises(AccessPointException, match="Could not find etype with name x"):
        fap.check_resource("x", "etype", access_point=ap)


def test_check_resource_reports_wrong_category(ap, forge):
    ap._available_etypes = []
    forge.ontology["x"] = Resource(id="other")
    with pytest.raises(AccessPointException, match="Resource x is not a etype"):
        fap.check_resource("x", "etype", access_point=ap)


def test_check_resource_builds_ontology_access_point(nexus):
    with pytest.raises(AccessPointException, match="Could not find"):
        fap.check_resource("x", "etype", access_token=make_token())
    assert nexus.ontology.resolve_calls[0]["text"] == "x"


def test_get_available_traces_searches_all_trace_projects(nexus):
    nwb = {"a.nwb": b""}
    first = nexus.forges["bbp/lnmce"].add("ExperimentalTrace", files=nwb, subject="mouse")
    second = nexus.forges["public/thalamus"].add("ExperimentalTrace", files=nwb, subject="mouse")
    nexus.forges["bbp/lnmce"].add("ExperimentalTrace", files=nwb, subject="rat")
    nexus.forges["bbp/lnmce"].add("ExperimentalTrace", files={"a.abf": b""}, subject="mouse")
    traces = fap.get_available_traces(species="mouse", access_token=make_token())
    assert traces == [first, second]


def test_get_available_traces_filters_on_brain_region(nexus):
    nexus.traces.add("ExperimentalTrace", files={"a.nwb": b""}, brainLocation="SS")
    assert not fap.get_available_traces(brain_region="CA1", access_token=make_token())
    assert fap.get_available_traces(brain_region="SS", access_token=make_token())


@pytest.mark.parametrize(
    "requested, known",
    [
        ("SSCX", "somatosensory areas"),
        ("sscx", "somatosensory areas"),
        ("all", "Basic cell groups and regions"),
        ("isocortex", "isocortex"),
        ("isocortex", "Isocortex"),
        ("Isocortex", "isocortex"),
        ("x", "X"),
    ],
)
def test_get_brain_region_tries_alternative_spellings(nexus, requested, known):
    region = Resource(id="br", label=known)
    nexus.ontology.ontology[known] = region
    assert fap.get_brain_region(requested, access_token=make_token()) is region


def test_get_brain_region_takes_first_of_several(nexus):
    region = Resource(id="br")
    nexus.ontology.ontology["CA1"] = [region, Resource(id="other")]
    assert fap.get_brain_region("CA1", access_token=make_token()) is region


def test_get_brain_region_unknown(nexus):
    with pytest.raises(AccessPointException, match="Could not find any brain region"):
        fap.get_brain_region("", access_token=make_token())


def test_brain_region_helpers(nexus):
    token = make_token()
    nexus.ontology.ontology["CA1"] = Resource(id="br", label="CA1", notation="CA1sp")
    expected = {"id": "br", "label": "CA1"}
    assert fap.get_brain_region_dict("CA1", access_token=token) == expected
    assert fap.get_brain_region_notation("CA1", access_token=token) == "CA1sp"
    assert fap.get_nexus_brain_region("CA1", access_token=token) == {
        "type": "BrainLocation",
        "brainRegion": expected,
    }
    assert fap.get_brain_region_notation(None) is None
    assert fap.get_nexus_brain_region(None) is None


def test_get_all_species_is_sorted_and_unique(nexus):
    for label in ("Rat", "Mouse", "Rat"):
        nexus.ontology.add("Class", subClassOf="nsg:Species", label=label)
    assert fap.get_all_species(access_token=make_token()) == ["Mouse", "Rat"]


def quality(label):
    return {"type": ["QualityAnnotation"], "hasBody": {"label": label}}


@pytest.mark.parametrize("wrap", [lambda a: a, lambda a: [a]])
def test_get_curated_morphology_prefers_curated(wrap):
    plain = Resource.from_json({"annotation": wrap(quality("Draft"))})
    curated = Resource.from_json({"annotation": wrap(quality("Curated"))})
    assert fap.get_curated_morphology([plain, curated]) is curated


def test_get_curated_morphology_accepts_derived():
    derived = Resource(derivation="x")
    assert fap.get_curated_morphology([Resource(), derived]) is derived


def test_get_curated_morphology_ignores_other_annotations():
    other = Resource.from_json({"annotation": {"type": "MTypeAnnotation"}})
    assert fap.get_curated_morphology([other]) is None


def located(label):
    return Resource.from_json({"brainLocation": {"brainRegion": {"label": label}}})


def test_filter_mechanisms_direct_match(forge):
    match = located("CA1")
    assert fap.filter_mechanisms_with_brain_region(
        forge, [match, located("SS"), Resource()], "CA1", set()
    ) == ([match], {"CA1"})
    assert not forge.sparql_calls


def test_filter_mechanisms_climbs_is_part_of(forge):
    match = located("Cortex")
    forge.sparql_answers = [(('"Layer 5"', "isPartOf"), [Resource(label="Cortex")])]
    assert fap.filter_mechanisms_with_brain_region(forge, [match], "Layer 5", set()) == (
        [match],
        {"Layer 5", "Cortex"},
    )


def test_filter_mechanisms_climbs_is_layer_part_of(forge):
    match = located("Cortex")
    forge.sparql_answers = [(('"Layer 5"', "isLayerPartOf"), [Resource(label="Cortex")])]
    resources, visited = fap.filter_mechanisms_with_brain_region(forge, [match], "Layer 5", set())
    assert resources == [match]
    assert visited == {"Layer 5", "Cortex"}


def test_filter_mechanisms_no_match(forge):
    assert fap.filter_mechanisms_with_brain_region(forge, [located("SS")], "CA1", set()) == (
        None,
        {"CA1"},
    )


def test_filter_mechanisms_tries_every_layer_parent(forge):
    match = located("Second")
    forge.sparql_answers = [
        (('"Layer 5"', "isLayerPartOf"), [Resource(label="First"), Resource(label="Second")])
    ]
    resources, visited = fap.filter_mechanisms_with_brain_region(forge, [match], "Layer 5", set())
    assert resources == [match]
    assert visited == {"Layer 5", "First", "Second"}


def test_filter_mechanisms_stops_on_hierarchy_cycle(forge):
    forge.sparql_answers = [
        (('"A"', "isPartOf"), [Resource(label="B")]),
        (('"B"', "isPartOf"), [Resource(label="A")]),
    ]
    assert fap.filter_mechanisms_with_brain_region(forge, [], "A", set()) == (None, {"A", "B"})
