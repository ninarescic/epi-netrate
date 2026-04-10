from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epinet.baseline import infer_netrate_baseline


def main() -> None:
    cascades_path = ROOT / "outputs" / "higgs_rt_cascades_21_50_top50.csv"
    out_path = ROOT / "outputs" / "higgs_rt_inferred_21_50_top50_10T.csv"

    df = pd.read_csv(cascades_path)

    df["infection_time"] = (
            df["infection_time"]
            - df.groupby("cascade_id")["infection_time"].transform("min")
    )

    relative_cascades_path = Path(cascades_path).with_name(
        Path(cascades_path).stem + "_relative.csv"
    )

    df.to_csv(relative_cascades_path, index=False)

    T = df["infection_time"].max()
    print("Max infection_time T =", T)

    infer_netrate_baseline(
        cascades_path=relative_cascades_path,
        out_path=out_path,
        params={
            "matlab_faithful": True,
            "horizon": 10 * T,
            "type_diffusion": "exp",
            "thr": 1e-8,
        },
    )

    print("Saved inferred edges to:", out_path)


if __name__ == "__main__":
    main()