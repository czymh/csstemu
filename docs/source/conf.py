# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information
import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../../'))

project = 'CSST Emulator'
copyright = '2024, Zhao Chen' #(陈钊)
author = 'Zhao Chen'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary', 
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme', # for readthedocs theme
]

autodoc_default_options = {
    'private-members': False,
    'special-members': '__init__',
    'member-order'   : 'bysource',
    # other options...
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'version_selector': True,
    'language_selector': True,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
html_context = {}
