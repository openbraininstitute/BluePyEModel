"""Nexus endpoint URLs.

Kept apart from forge_access_point so that code which only needs the URLs does not
import the optional Nexus dependencies (kgforge, entity_management, jwt).
"""

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

DEFAULT_NEXUS_ENDPOINT = "https://openbluebrain.com/api/nexus/v1"
STAGING_NEXUS_ENDPOINT = "https://staging.openbluebrain.com/api/nexus/v1"
