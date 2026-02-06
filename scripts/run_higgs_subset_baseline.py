from pathlib import Path

from epinet.paths import project_root
from epinet.baseline import infer_netrate_baseline


def main():
    root = project_root()
    cascades = root / "data" / "processed" / "higgs_rt_cascades_tiny.csv"
    out = root / "outputs" / "higgs_rt_subset_inferred.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not cascades.exists():
        raise FileNotFoundError(f"Missing {cascades}. Run scripts/make_higgs_subset_cascades.py first.")

    infer_netrate_baseline(cascades, out)
    print("Done. Inferred:", out)


if __name__ == "__main__":
    main()
