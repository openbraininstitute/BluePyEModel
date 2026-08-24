# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../."))

from importlib.metadata import version


# -- Project information -----------------------------------------------------

project = "BluePyEModel"
author = "Blue Brain Project/EPFL"

# The short X.Y version
version = version("bluepyemodel")

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
]

suppress_warnings = ["docutils", "autodoc.mocked_object"]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_autosummary/bluepyemodel.icselector.met_type_ic_profile_generator.rst"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "obi_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
doctest_global_setup = """
from bluepyemodel.tools.utils import (
    get_curr_name,
    get_loc_name,
    get_protocol_name,
    parse_feature_name_parts,
)
"""
# html_static_path = ['_static']

html_title = "BluePyEModel"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# autosummary settings
autosummary_generate = True

# autodoc settings
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autoclass_content = "both"

add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "luigi": ("https://luigi.readthedocs.io/en/stable", None),
}

autodoc_mock_imports = [
    "bluepyemodel.tasks.luigi_tools",
    "bluepyemodel.icselector.met_type_ic_profile_generator",
    "entity_management",
    "jwt",
    "kgforge",
]
