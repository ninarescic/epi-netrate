from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    src = ROOT / "outputs" / "higgs_rt_cascades_all.csv"
    dst = ROOT / "outputs" / "higgs_rt_cascades_21_50_top50.csv"

    df = pd.read_csv(src)

    sizes = df.groupby("cascade_id")["node_id"].nunique()
    eligible = sizes[(sizes >= 21) & (sizes <= 50)].sort_values(ascending=False)
    kept_ids = eligible.head(50).index

    out = df[df["cascade_id"].isin(kept_ids)].copy()
    out = out.sort_values(["cascade_id", "infection_time", "node_id"]).reset_index(drop=True)

    out.to_csv(dst, index=False)
    print("Saved:", dst)
    print("cascades kept:", out["cascade_id"].nunique())
    print("rows:", len(out))

if __name__ == "__main__":
    main()