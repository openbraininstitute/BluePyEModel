"""The local pipeline must not need the optional Nexus dependencies."""

"""
Copyright 2025, Open Brain Institute

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

import subprocess
import sys
import textwrap

import pytest

# CI installs every extra, so block the Nexus packages to mimic a base install.
BLOCK_NEXUS = textwrap.dedent("""
    import sys
    for name in ("kgforge", "entity_management", "jwt"):
        sys.modules[name] = None
    """)


def run_without_nexus(code):
    """Run code in a fresh interpreter in which the Nexus packages cannot be imported."""
    return subprocess.run(
        [sys.executable, "-c", BLOCK_NEXUS + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "statement",
    [
        "from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline",
        "from bluepyemodel.access_point import get_access_point",
        "from bluepyemodel.access_point.local import LocalAccessPoint",
        "from bluepyemodel.access_point.nexus_endpoints import DEFAULT_NEXUS_ENDPOINT",
    ],
)
def test_local_pipeline_imports_without_nexus(statement):
    result = run_without_nexus(statement)
    assert result.returncode == 0, result.stderr


def test_nexus_access_point_still_needs_nexus():
    """Check that the blocking works, so the test above cannot pass vacuously."""
    result = run_without_nexus("import bluepyemodel.access_point.forge_access_point")
    assert result.returncode != 0
    assert "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr


def test_forge_access_point_reexports_default_endpoint():
    """Code importing the endpoint from forge_access_point keeps working."""
    pytest.importorskip("kgforge")
    from bluepyemodel.access_point import forge_access_point
    from bluepyemodel.access_point import nexus_endpoints

    assert forge_access_point.DEFAULT_NEXUS_ENDPOINT == nexus_endpoints.DEFAULT_NEXUS_ENDPOINT
