from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Set

import os
import pandas as pd

# Headless-safe matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epinet.paths import project_root


def get_higgs_dir() -> Path:
    env = os.getenv("HIGGS_DIR")
    if not env:
        raise RuntimeError("HIGGS_DIR is not set. Example: export HIGGS_DIR=/data/higgs-twitter")
    return Path(env).expanduser()


def load_rt_proxy_edges(higgs_dir: Path) -> Set[Tuple[str, str]]:
    """
    Loads higgs-retweet_network.edgelist into a set of directed edges (u, v) as strings.
    """
    path = higgs_dir / "higgs-retweet_network.edgelist"
    if not path.exists():
        raise FileNotFoundError(f"Missing proxy RT graph: {path}")

    edges: Set[Tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            edges.add((str(u), str(v)))
    return edges


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def detect_inferred_schema(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """
    Try to infer which columns are: src, dst, weight.
    """
    cols = list(df.columns)
    src = pick_col(cols, ["src", "source", "u", "from", "i", "node_u", "node_i"])
    dst = pick_col(cols, ["dst", "target", "v", "to", "j", "node_v", "node_j"])
    if src is None or dst is None:
        if len(cols) < 2:
            raise ValueError(f"Need at least 2 columns for edges, got: {cols}")
        src = src or cols[0]
        dst = dst or cols[1]
    w = pick_col(cols, ["weight", "rate", "alpha", "score", "w", "value", "lambda"])
    return src, dst, w


def precision_at_k(pred: List[Tuple[str, str]], truth: Set[Tuple[str, str]], k: int) -> float:
    topk = pred[:k]
    if not topk:
        return 0.0
    hits = sum(1 for e in topk if e in truth)
    return hits / len(topk)


def to_undirected(e: Tuple[str, str]) -> Tuple[str, str]:
    a, b = e
    return (a, b) if a <= b else (b, a)


def main():
    root = project_root()
    higgs_dir = get_higgs_dir()

    inferred_path = root / "outputs" / "higgs_rt_subset_inferred.csv"
    if not inferred_path.exists():
        raise FileNotFoundError(f"Missing inferred file: {inferred_path}")

    # Load inferred edges
    df = pd.read_csv(inferred_path)
    if df.empty:
        raise ValueError(f"{inferred_path} is empty.")

    src_col, dst_col, w_col = detect_inferred_schema(df)
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)

    if w_col is not None:
        df[w_col] = pd.to_numeric(df[w_col], errors="coerce")
        df = df.sort_values(by=w_col, ascending=False, na_position="last")

    inferred_edges = list(zip(df[src_col].tolist(), df[dst_col].tolist()))

    # Load proxy RT network
    rt_edges = load_rt_proxy_edges(higgs_dir)

    # For direction ambiguity, compute:
    # (A) directed match
    # (B) undirected match (edge exists regardless of direction)
    rt_undirected = {to_undirected(e) for e in rt_edges}
    inferred_undirected = [to_undirected(e) for e in inferred_edges]

    # Evaluate
    K = min(2000, len(inferred_edges))
    ks = list(range(1, K + 1))
    p_dir = [precision_at_k(inferred_edges, rt_edges, k) for k in ks]
    p_undir = [precision_at_k(inferred_undirected, rt_undirected, k) for k in ks]

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a small metrics CSV
    metrics_csv = out_dir / "higgs_rt_proxy_precision_at_k.csv"
    pd.DataFrame({"k": ks, "precision_directed": p_dir, "precision_undirected": p_undir}).to_csv(
        metrics_csv, index=False
    )

    # Plot
    fig_path = out_dir / "higgs_rt_proxy_precision_at_k.png"
    plt.figure()
    plt.plot(ks, p_dir, label="directed")
    plt.plot(ks, p_undir, label="undirected")
    plt.xlabel("k")
    plt.ylabel("precision@k")
    plt.title("Higgs: inferred edges vs RT proxy graph")
    plt.legend()
    plt.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close()

    print("Inferred file:", inferred_path)
    print("Proxy RT graph:", higgs_dir / "higgs-retweet_network.edgelist")
    print("Detected inferred columns:", {"src": src_col, "dst": dst_col, "weight": w_col})
    print("Saved:", metrics_csv)
    print("Saved:", fig_path)

    # Quick console summary
    for k in [10, 50, 100, 500, 1000, 2000]:
        if k <= K:
            print(f"k={k:>4}  precision_directed={p_dir[k-1]:.4f}  precision_undirected={p_undir[k-1]:.4f}")


if __name__ == "__main__":
    main()
