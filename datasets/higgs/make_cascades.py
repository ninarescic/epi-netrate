from __future__ import annotations

from pathlib import Path
import gzip
import pandas as pd


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt", encoding="utf-8")


def _find_activity_file(higgs_dir: Path) -> Path:
    candidates = [
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
    ]
    for p in candidates:
        if p.exists():
            return p

    gl = list(higgs_dir.glob("*activity_time*"))
    if gl:
        return gl[0]

    raise FileNotFoundError(f"Could not find activity file in {higgs_dir}")


def make_interaction_cascades(
    higgs_dir: Path,
    out_csv: Path,
    *,
    min_cascade_size: int = 10,
    limit_rows: int | None = None,
    max_cascades: int | None = None,
    allowed_kinds: set[str] | None = None,
) -> Path:
    activity = _find_activity_file(higgs_dir)

    if allowed_kinds is None:
        allowed_kinds = {"RT", "MT", "RE"}

    rows = []

    with _open_maybe_gzip(activity) as f:
        for idx, line in enumerate(f):
            if limit_rows is not None and idx >= limit_rows:
                break

            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                continue

            a, b, ts, kind = parts

            if kind not in allowed_kinds:
                continue

            rows.append((b, a, float(ts), kind))

    df = pd.DataFrame(
        rows,
        columns=["cascade_id", "node_id", "infection_time", "interaction_type"],
    )

    if df.empty:
        raise RuntimeError(
            f"No rows found in activity log for allowed_kinds={sorted(allowed_kinds)}."
        )

    # Add synthetic seed at earliest observed time per cascade
    t0 = df.groupby("cascade_id", as_index=False)["infection_time"].min()
    t0["node_id"] = t0["cascade_id"]
    t0["interaction_type"] = "SEED"
    seeds = t0[["cascade_id", "node_id", "infection_time", "interaction_type"]]

    df = pd.concat([df, seeds], ignore_index=True)

    # Filter by cascade size
    sizes = df.groupby("cascade_id")["node_id"].nunique()
    keep = sizes[sizes >= min_cascade_size].index
    df = df[df["cascade_id"].isin(keep)].copy()

    # Keep largest cascades, not first lexicographic ones
    if max_cascades is not None:
        sizes = df.groupby("cascade_id")["node_id"].nunique().sort_values(ascending=False)
        kept_ids = list(sizes.head(max_cascades).index)
        df = df[df["cascade_id"].isin(kept_ids)].copy()

    df = df.sort_values(
        ["cascade_id", "infection_time", "node_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def make_rt_cascades(
    higgs_dir: Path,
    out_csv: Path,
    *,
    min_cascade_size: int = 10,
    limit_rows: int | None = None,
    max_cascades: int | None = None,
) -> Path:
    return make_interaction_cascades(
        higgs_dir=higgs_dir,
        out_csv=out_csv,
        min_cascade_size=min_cascade_size,
        limit_rows=limit_rows,
        max_cascades=max_cascades,
        allowed_kinds={"RT"},
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    higgs_dir = root / "data" / "higgs"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("higgs_rt_cascades_all.csv", {"RT"}),
        ("higgs_rt_re_cascades.csv", {"RT", "RE"}),
        ("higgs_rt_mt_cascades.csv", {"RT", "MT"}),
        ("higgs_rt_re_mt_cascades.csv", {"RT", "RE", "MT"}),
    ]

    for filename, allowed_kinds in configs:
        out_csv = out_dir / filename
        out = make_interaction_cascades(
            higgs_dir=higgs_dir,
            out_csv=out_csv,
            min_cascade_size=10,
            max_cascades=None,
            allowed_kinds=allowed_kinds,
        )
        print(f"Saved {sorted(allowed_kinds)} cascades to: {out}")