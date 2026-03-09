project = "DeepExtractor"
author = "Tom Dooney"
copyright = "2025, Tom Dooney"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # NumPy/Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "myst_parser",               # Markdown support
    "nbsphinx",                  # Render Jupyter notebooks
]

autoapi_dirs = ["../src"]
autoapi_type = "python"
autoapi_options = ["members", "undoc-members", "show-inheritance"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Do not re-execute notebooks during docs build
nbsphinx_execute = "never"
