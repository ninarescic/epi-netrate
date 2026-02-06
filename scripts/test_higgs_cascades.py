from pathlib import Path
import pandas as pd

from epinet.datasets import get_higgs_dir
from epinet.paths import project_root
from datasets.higgs.make_cascades import make_rt_cascades

def main():
    higgs_dir = get_higgs_dir()
    print("Using Higgs dir:", higgs_dir)

    out = project_root() / "data" / "processed" / "higgs_rt_cascades_tiny.csv"

    make_rt_cascades(
        higgs_dir=higgs_dir,
        out_csv=out,
        min_cascade_size=5,     # smaller = more cascades survive on tiny sample
        limit_rows=20_000,      # VERY small read
        max_cascades=50,        # keep only 50 cascades
    )

    df = pd.read_csv(out)
    print("Wrote:", out)
    print("Rows:", len(df), "Cascades:", df["cascade_id"].nunique(), "Nodes:", df["node_id"].nunique())
    print(df.head())

if __name__ == "__main__":
    main()
