from pathlib import Path
import pandas as pd

from epinet.baseline import infer_netrate_baseline
from epinet.paths import project_root


def ensure_schema(path: Path) -> None:
    df = pd.read_csv(path)

    # Accept older names but rewrite to what legacy expects
    if "node_id" not in df.columns and "node" in df.columns:
        df = df.rename(columns={"node": "node_id"})
    if "infection_time" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "infection_time"})

    required = {"cascade_id", "node_id", "infection_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Columns are: {list(df.columns)}")

    df.to_csv(path, index=False)


def main():
    cascades = project_root() / "data" / "processed" / "toy_cascades.csv"
    out = project_root() / "outputs" / "toy_inferred.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not cascades.exists():
        raise FileNotFoundError(f"Missing {cascades}. Run scripts/make_toy_cascades.py first.")

    ensure_schema(cascades)

    print("Using cascades file:", cascades.resolve())
    print(pd.read_csv(cascades).head())

    infer_netrate_baseline(cascades, out)
    print("Done. Output at:", out)

if __name__ == "__main__":
    main()
