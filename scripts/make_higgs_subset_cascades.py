from pathlib import Path
import pandas as pd


from epinet.datasets import get_higgs_dir
from epinet.paths import project_root
from datasets.higgs.make_cascades import make_rt_cascades
import os
print("DEBUG HIGGS_DIR =", os.getenv("HIGGS_DIR"))

def main():
    root = project_root()
    higgs_dir = get_higgs_dir()
    print("Using Higgs dir:", higgs_dir)

    tmp = root / "data" / "processed" / "higgs_rt_tmp.csv"
    out = root / "data" / "processed" / "higgs_rt_subset.csv"

    # Build cascades from limited activity rows (fast)
    make_rt_cascades(
        higgs_dir=higgs_dir,
        out_csv=tmp,
        min_cascade_size=10,
        limit_rows=200_000,  # adjust up later
    )

    # Optional: further reduce by keeping only first N cascades
    df = pd.read_csv(tmp)
    keep_cascades = df["cascade_id"].drop_duplicates().head(200).tolist()
    df = df[df["cascade_id"].isin(keep_cascades)].copy()
    df.to_csv(out, index=False)

    print("Wrote subset cascades:", out)
    print("Events:", len(df), "Cascades:", df["cascade_id"].nunique(), "Nodes:", df["node_id"].nunique())
    print(df.head())


if __name__ == "__main__":
    main()
