from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List


_SUPPORTED_PARAMS = {
    "l1",
    "thr",
    "topk",
    "solver",
    "seed",
    "save_B",
    "save_nodes",
    "no_self_loops",
    "matlab_faithful",
    "horizon",
    "type_diffusion",
}


def infer_netrate_baseline(
    cascades_path: str | Path,
    out_path: str | Path,
    *,
    candidates_path: Optional[str | Path] = None,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run the legacy NetRate implementation through a stable function API
    by patching sys.argv.
    """
    cascades_path = str(Path(cascades_path))
    out_path = str(Path(out_path))
    params = params or {}

    if candidates_path is not None:
        raise NotImplementedError(
            "candidates_path is not supported by netrate_legacy.netrate_infer yet."
        )

    unknown = sorted(set(params) - _SUPPORTED_PARAMS)
    if unknown:
        raise ValueError(
            f"Unsupported params for netrate_infer: {unknown}. "
            f"Supported params: {sorted(_SUPPORTED_PARAMS)}"
        )

    argv: List[str] = []
    argv += ["--cascades", cascades_path]
    argv += ["--out", out_path]

    for k, v in params.items():
        flag = f"--{k}"
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