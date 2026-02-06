from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    # epinet/paths.py -> epinet -> project root
    return Path(__file__).resolve().parents[1]
