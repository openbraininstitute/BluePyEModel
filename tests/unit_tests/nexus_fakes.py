"""In-memory stand-in for Nexus Forge, shared by the synthetic Nexus access point tests.

Only ``KnowledgeGraphForge`` is replaced. Token handling, filter construction, JSON dumps and the
real ``kgforge`` ``Resource``/``Dataset`` classes all still run, so tests can assert on what ends
up stored rather than on which mock was called. Nothing here touches the network.
"""

import json
import pathlib
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from types import SimpleNamespace

import jwt
from kgforge.core import Resource
from kgforge.core.commons.actions import LazyAction


def make_token(lifetime_s=3600, username="test_user"):
    """Return a genuine (unverified) JWT expiring ``lifetime_s`` seconds from now."""
    expiry = datetime.now(timezone.utc) + timedelta(seconds=lifetime_s)
    payload = {
        "preferred_username": username,
        "name": "Test User",
        "email": "test_user@example.com",
        "sub": "test_sub",
        "exp": int(expiry.timestamp()),
    }
    return jwt.encode(payload, "unit-test-secret-of-a-sufficient-length", algorithm="HS256")


def to_json(value):
    """Convert resources to plain containers, the way ``forge.as_json`` does."""
    if isinstance(value, Resource):
        return {k: to_json(v) for k, v in vars(value).items() if not k.startswith("_")}
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    return value


def matches(document, filters):
    """Return True if ``document`` satisfies the nested ``filters``."""
    for key, expected in filters.items():
        actual = document.get(key)
        if isinstance(actual, list) and key != "type":
            if not any(matches({key: item}, {key: expected}) for item in actual):
                return False
        elif key == "type":
            if expected not in (actual if isinstance(actual, list) else [actual]):
                return False
        elif isinstance(expected, dict):
            if not isinstance(actual, dict) or not matches(actual, expected):
                return False
        elif actual != expected:
            return False
    return True


def _execute_lazy(value):
    if isinstance(value, LazyAction):
        return value.execute()
    if isinstance(value, list):
        return [_execute_lazy(item) for item in value]
    return value


class FakeForge:
    """One Nexus project (bucket), held in memory."""

    def __init__(self):
        self.resources = []
        self.blobs = {}
        self.ontology = {}
        self.suggestions = {}
        self.sparql_answers = []
        self.search_calls = []
        self.resolve_calls = []
        self.sparql_calls = []
        self.retrieve_calls = []
        self.schema_ids = []
        self.updated = []
        self.deprecated = []
        self._debug = True  # let kgforge's ``@catch`` re-raise instead of printing errors
        self._model = SimpleNamespace(schema_id=lambda type_: f"schema:{type_}")
        self._store = SimpleNamespace(upload_image=self._upload_image)
        self._next_id = 0

    # Helpers for tests

    def add(self, type_, payload=None, files=None, id_=None, **fields):
        """Seed a resource, optionally with a JSON payload and files as distributions."""
        document = {"id": id_ or self._new_id(), "type": ["Entity", type_], **fields}
        document.setdefault("name", document["id"].replace(":", "_"))
        files = dict(files or {})
        if payload is not None:
            files[f"{document['name']}.json"] = json.dumps(payload).encode()
        distributions = [self._add_blob(name, content) for name, content in files.items()]
        if distributions:
            document["distribution"] = (
                distributions[0] if len(distributions) == 1 else distributions
            )
        return self._save(document)

    def active(self, type_):
        """Return the non-deprecated resources of a given type."""
        return [r for r in self.resources if matches(to_json(r), {"type": type_})]

    def payload_of(self, resource):
        """Return the JSON payload attached to a stored resource."""
        distributions = resource.distribution
        if not isinstance(distributions, list):
            distributions = [distributions]
        for distribution in distributions:
            if distribution.name.endswith(".json"):
                return json.loads(self.blobs[distribution.contentUrl])
        raise AssertionError(f"No JSON distribution on {resource}")

    def _new_id(self):
        self._next_id += 1
        return f"fake:{self._next_id}"

    def _add_blob(self, name, content):
        url = f"blob://{self._new_id()}"
        self.blobs[url] = content
        suffix = pathlib.Path(name).suffix[1:]
        return {"name": name, "contentUrl": url, "encodingFormat": f"application/{suffix}"}

    def _save(self, document):
        self.resources = [r for r in self.resources if r.id != document["id"]]
        resource = Resource.from_json(document)
        resource._store_metadata = {"_rev": 1, "_deprecated": False}
        self.resources.append(resource)
        return resource

    def _upload(self, path, content_type):
        blob = self._add_blob(pathlib.Path(path).name, pathlib.Path(path).read_bytes())
        blob["encodingFormat"] = content_type
        return Resource.from_json(blob)

    def _upload_image(self, path, content_type, about, forge):
        return Resource.from_json(
            {"name": pathlib.Path(path).name, "encodingFormat": content_type, "about": about}
        )

    def _persist(self, resource):
        for attribute in ("distribution", "image"):
            if hasattr(resource, attribute):
                setattr(resource, attribute, _execute_lazy(getattr(resource, attribute)))
        document = to_json(resource)
        document.setdefault("id", self._new_id())
        return self._save(document)

    # The subset of the KnowledgeGraphForge API used by BluePyEModel

    def from_json(self, data, na=None):
        return Resource.from_json(data, na)

    def as_json(self, resource):
        return to_json(resource)

    def reshape(self, resource, keep, versioned=False):
        kept = {k: getattr(resource, k) for k in keep if hasattr(resource, k) and k[0] != "_"}
        return Resource(**kept)

    def attach(self, path, content_type=None):
        return LazyAction(self._upload, path, content_type)

    def register(self, resource, schema_id=None):
        self.schema_ids.append(schema_id)
        return self._persist(resource)

    def update(self, resource, schema_id=None):
        self.schema_ids.append(schema_id)
        self.updated.append(resource.id)
        return self._persist(resource)

    def deprecate(self, resource):
        self.deprecated.append(resource.id)
        self.resources = [r for r in self.resources if r.id != resource.id]

    def retrieve(self, id=None, cross_bucket=False, **_):  # pylint: disable=redefined-builtin
        self.retrieve_calls.append(id)
        return next((r for r in self.resources if r.id == id), None)

    def search(self, filters, **kwargs):
        self.search_calls.append({"filters": filters, **kwargs})
        return [r for r in self.resources if matches(to_json(r), filters)]

    def resolve(self, text, scope=None, strategy=None, limit=None):
        self.resolve_calls.append({"text": text, "scope": scope, "strategy": strategy})
        if limit is not None and limit > 1:
            return self.suggestions.get(text)
        return self.ontology.get(text)

    def sparql(self, query, limit=None):
        self.sparql_calls.append(query)
        for needles, answer in self.sparql_answers:
            if all(needle in query for needle in needles):
                return answer
        return None

    def download(self, resource, path, directory, **_):
        distributions = resource.distribution
        if not isinstance(distributions, list):
            distributions = [distributions]
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        for distribution in distributions:
            if distribution.contentUrl in self.blobs:
                target = pathlib.Path(directory) / distribution.name
                target.write_bytes(self.blobs[distribution.contentUrl])


# kgforge's ``@catch`` decorator finds the forge attached to a Dataset by this exact class name.
FakeForge.__name__ = "KnowledgeGraphForge"


class FakeNexus:
    """A set of Nexus projects; ``connect`` replaces ``KnowledgeGraphForge``."""

    def __init__(self):
        self.forges = defaultdict(FakeForge)
        self.connections = []

    def connect(self, forge_path, bucket=None, endpoint=None, token=None):
        self.connections.append(
            SimpleNamespace(forge_path=forge_path, bucket=bucket, endpoint=endpoint, token=token)
        )
        return self.forges[bucket]

    @property
    def project(self):
        return self.forges["demo/emodel"]

    @property
    def ontology(self):
        return self.forges["neurosciencegraph/datamodels"]

    @property
    def atlas(self):
        return self.forges["bbp/atlas"]

    @property
    def traces(self):
        return self.forges["bbp/lnmce"]
