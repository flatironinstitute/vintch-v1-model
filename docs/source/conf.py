# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'vintch_v1_model'
copyright = '2025, NeuroRSE'
author = 'NeuroRSE'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # latexsyntax for math
    "sphinx.ext.mathjax",
    # docstrings
    "sphinx.ext.napoleon",
    "numpydoc",
    # matplotlib in your docs
    "matplotlib.sphinxext.plot_directive",
    "matplotlib.sphinxext.mathmpl",
    # build API documentation
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.apidoc",
    # link to other sphinx projects (and maybe your own?)
    "sphinx.ext.intersphinx",
    # copy code easily
    "sphinx_copybutton",
    # markdown and text-based notebooks
    "myst_nb",
    # bibtex supports
    "sphinxcontrib.bibtex",
    # little button to view code
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# MYST

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# SPHINXCONTRIB-BIBTEX
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# SPHINX COPYBUTTON
copybutton_exclude = ".linenos, .gp"

# APIDOC
apidoc_module_dir = "../../src/vintch_model/"
