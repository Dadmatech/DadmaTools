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
sys.path.insert(0, os.path.abspath('..'))

import sphinx_markdown_parser
from sphinx_markdown_parser.parser import MarkdownParser
from sphinx_markdown_parser.transform import AutoStructify

# -- Project information -----------------------------------------------------

project = 'Dadmatools'
copyright = '2021, Dadmatech Company'
author = 'Dadmatech Company'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx_markdown_tables',
        'sphinx.ext.todo',
        'sphinx.ext.autosectionlabel',
        'sphinx.ext.napoleon',
        'sphinx.ext.mathjax'
]

source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
        'logo_only': True,
        'display_version': True,
        'prev_next_buttons_location': 'bottom',
        'style_external_links': True,
        # Toc options
        'collapse_navigation': False,
        'sticky_navigation': False,
        'navigation_depth': 3,
        'includehidden': False,
        'titles_only': False
}

html_title = "DadmaTools documentation"
# html_logo = "docs/imgs/dadmatools_logo.png"
# html_favicon = "docs/imgs/dadmatools_logo.png"
#html_style = 'custom.css'

master_doc = 'index'

github_doc_root = 'https://github.com/Dadmatech/DadmaTools/tree/master/readthedocs/docs'


autosectionlabel_prefix_document = True

def setup(app):
    app.add_source_suffix('.md', 'markdown')
    app.add_source_parser(MarkdownParser)
    app.add_config_value('markdown_parser_config', {
        "extensions": ["tables","extra"],
        'auto_toc_tree_section': 'Content',
        'enable_auto_toc_tree': True,
        'enable_eval_rst': True,
        'enable_inline_math': True,
        'enable_math': True
    }, True)
    app.add_transform(AutoStructify)
