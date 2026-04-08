from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epinet.baseline import infer_netrate_baseline


def main() -> None:
    cascades_path = ROOT / "outputs" / "higgs_rt_cascades_21_50_top50.csv"
    out_path = ROOT / "outputs" / "higgs_rt_inferred_21_50_top50.csv"

    infer_netrate_baseline(
        cascades_path=cascades_path,
        out_path=out_path,
        params={
            "matlab_faithful": True,
            "horizon": 10.0,
            "type_diffusion": "exp",
            "thr": 1e-8,
        },
    )


    print("Saved inferred edges to:", out_path)


if __name__ == "__main__":
    main()