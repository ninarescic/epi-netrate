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
        higgs_dir / "higgs-activity_time.txt.gz",  # keep
        higgs_dir / "higgs-activity_time.txt",     # keep
        higgs_dir / "higgs-activity_time.txt",     # your case
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        # SNAP default name too:
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt",
        # also support SNAP’s original:
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
        higgs_dir / "higgs-activity_time.txt",
        # simplest:
        higgs_dir / "higgs-activity_time.txt",
        # SNAP page original:
        higgs_dir / "higgs-activity_time.txt.gz",
        higgs_dir / "higgs-activity_time.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # also try a glob
    gl = list(higgs_dir.glob("*activity_time*"))
    if gl:
        return gl[0]
    raise FileNotFoundError(f"Could not find activity file in {higgs_dir}")


def make_rt_cascades(
    higgs_dir: Path,
    out_csv: Path,
    *,
    min_cascade_size: int = 10,
    limit_rows: int | None = None,
    max_cascades: int | None = None,
) -> Path:
    activity = _find_activity_file(higgs_dir)

    rows = []
    with _open_maybe_gzip(activity) as f:
        for idx, line in enumerate(f):
            if limit_rows is not None and idx >= limit_rows:
                break
            line = line.strip()
            if not line:
                continue
            a, b, ts, kind = line.split()
            if kind != "RT":
                continue
            rows.append((b, a, float(ts)))  # cascade_id=b, node_id=a

    df = pd.DataFrame(rows, columns=["cascade_id", "node_id", "infection_time"])
    if df.empty:
        raise RuntimeError("No RT rows found in activity log.")

    # add seed
    t0 = df.groupby("cascade_id")["infection_time"].min().reset_index()
    t0["node_id"] = t0["cascade_id"]
    seeds = t0[["cascade_id", "node_id", "infection_time"]]
    df = pd.concat([df, seeds], ignore_index=True)

    sizes = df.groupby("cascade_id")["node_id"].nunique()
    keep = sizes[sizes >= min_cascade_size].index
    df = df[df["cascade_id"].isin(keep)].copy()

    # NEW: keep only the first max_cascades (sorted by cascade_id for determinism)
    if max_cascades is not None:
        kept_ids = sorted(df["cascade_id"].unique())[:max_cascades]
        df = df[df["cascade_id"].isin(kept_ids)].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv

