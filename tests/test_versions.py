"""The package, pyproject.toml and compose.yaml must name the same version.

The release workflow publishes ``ghcr.io/<owner>/rag-ucdp:<version>`` from a
``v<version>`` tag, and compose.yaml pulls exactly that image.
"""

from __future__ import annotations

import re
import tomllib

from rag_ucdp import __version__
from rag_ucdp.config import REPO_ROOT


def test_versions_agree() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    compose = re.search(r"ghcr\.io/[^:\s]+:(\d[^\s\"}]*)", (REPO_ROOT / "compose.yaml").read_text())
    assert compose is not None, "compose.yaml must reference a ghcr.io image with a version tag"
    assert __version__ == pyproject == compose.group(1)
