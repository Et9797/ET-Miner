"""Sphinx configuration for the ET-Miner documentation."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Project metadata is read from pyproject.toml so it works without the package installed.
_pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
_repository = _pyproject["urls"]["Repository"]

project = _pyproject["name"]
release = _pyproject["version"]
version = ".".join(release.split(".")[:2])
author = ", ".join(a["name"] for a in _pyproject["authors"])
_copyright = re.search(r"Copyright (\d{4}) (.+?) \(", (ROOT / "LICENSE").read_text(encoding="utf-8"))
copyright = f"{_copyright.group(1)}, {_copyright.group(2)}" if _copyright else author

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "sphinx_llm.txt",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "**/.venv", "**/.ipynb_checkpoints"]

nitpicky = True
nitpick_ignore_regex = [
    # polars documents DataFrame/LazyFrame as method collections; its inventory
    # has no class entries, so type references to them cannot resolve.
    ("py:class", r"polars\..*"),
    # CuPy (imported as ``cp`` under TYPE_CHECKING) is an optional GPU
    # dependency and not part of intersphinx_mapping.
    ("py:class", r"(cupy|cp)\..*"),
]

# -- MyST ---------------------------------------------------------------------
# Heading anchors let pages link to README/CHANGELOG/LICENSE sections.
myst_heading_anchors = 3
# Resolve Markdown cross-references in these domains only; sphinx-argparse
# registers a domain that cannot take part in "any" lookups.
myst_ref_domains = ["std", "py"]

# -- API reference ------------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # HAS_GPU, HAS_RUST, ... are probed at import time; their value on the
    # docs builder says nothing about a user's machine.
    "no-value": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# "Returns:" lines in this code base are prose that may contain a colon; keep
# the text in the description instead of turning it into a type reference.
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
}

# -- HTML ---------------------------------------------------------------------
html_theme = "furo"
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_theme_options = {
    "source_repository": _repository,
    "source_branch": "main",
    "source_directory": "docs/",
}

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- llms.txt (sphinx_llm.txt) ------------------------------------------------
llms_txt_description = _pyproject["description"]
