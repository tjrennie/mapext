"""Configuration file for the Sphinx documentation builder.

This module sets up the Sphinx documentation for the 'mapext' project,
including project information, general configuration, and HTML output options.
For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mapext"
copyright = "2025, tjrennie"
author = "tjrennie"

try:
    from mapext import __version__ as release

    version = release.split("+")[0]
except (ImportError, ImportWarning):
    release = version = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "logo_only": True,
}
html_logo = "_static/mapext_logo.png"

# -- Autodoc configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__init__",
}
suppress_warnings = ["duplicate.object"]

# -- MyST Parser configuration -----------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/sphinx-extensions.html
myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "substitution",
    "tasklist",
]

# -- BibTeX configuration ---------------------------------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/
bibtex_bibfiles = ["_static/references.bib"]
bibtex_default_style = "author_year"
bibtex_cite_style = "author_year"
bibtex_reference_style = "author_year"
