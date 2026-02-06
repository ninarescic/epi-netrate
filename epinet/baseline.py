from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List


def infer_netrate_baseline(
    cascades_path: str | Path,
    out_path: str | Path,
    *,
    candidates_path: Optional[str | Path] = None,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run the legacy NetRate implementation (netrate_legacy/netrate_infer.py)
    through a stable function API by patching sys.argv.
    """
    cascades_path = str(Path(cascades_path))
    out_path = str(Path(out_path))
    candidates_path = str(Path(candidates_path)) if candidates_path else None
    params = params or {}

    # NOTE: The flag names below MUST match your argparse in netrate_infer.py.
    # If your script uses different names, we’ll change them in one place (here).
    argv: List[str] = []
    argv += ["--cascades", cascades_path]
    argv += ["--out", out_path]
    if candidates_path:
        argv += ["--candidates", candidates_path]

    for k, v in params.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

    from netrate_legacy import netrate_infer

    import sys
    old_argv = sys.argv[:]
    try:
        sys.argv = ["netrate_infer.py"] + argv
        netrate_infer.main()
    finally:
        sys.argv = old_argv
