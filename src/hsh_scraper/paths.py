"""Shared project paths for local scripts and package imports."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]


def _configured_dir(env_name: str, default: Path) -> Path:
    return Path(os.getenv(env_name, str(default))).expanduser()


DATA_DIR = _configured_dir("RAG_DATA_DIR", REPO_ROOT / "data")
ARTIFACTS_DIR = _configured_dir("RAG_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
CONFIG_DIR = _configured_dir("RAG_CONFIG_DIR", REPO_ROOT / "config")
