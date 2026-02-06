from __future__ import annotations

from pathlib import Path
import os


def get_higgs_dir() -> Path:
    """
    Returns the Higgs dataset directory from environment variable HIGGS_DIR.
    This is ideal for server runs where datasets live outside the repo.
    """
    env = os.getenv("HIGGS_DIR")
    if env:
        return Path(env).expanduser()

    raise RuntimeError(
        "HIGGS_DIR is not set. In your shell, run:\n"
        "  export HIGGS_DIR=~/dezinfo_data/higgs-twitter\n"
    )
