from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epinet.paths import project_root
from datasets.higgs.ground_truth import get_truth_edges


# =========================
# CONFIG: edit these values
# =========================

TRUTH_TYPE = ("social_1hop")
REVERSE_TRUTH = True
TOP_K_MAX = 2000

INFERRED_PATH = ROOT / "outputs" / "higgs_rt_inferred_21_50_top50_10T.csv"
HIGGS_DIR = ROOT / "data" / "higgs"


# =========================
# Helpers
# =========================
def detect_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Try to detect source, destination, and weight columns in inferred edge CSV.
    """

    cols = {c.lower(): c for c in df.columns}

    src_candidates = ["source", "src", "u", "parent", "from"]
    dst_candidates = ["target", "dst", "v", "child", "to"]
    weight_candidates = ["beta", "weight", "score", "alpha", "value"]

    src_col = next((cols[c] for c in src_candidates if c in cols), None)
    dst_col = next((cols[c] for c in dst_candidates if c in cols), None)
    w_col = next((cols[c] for c in weight_candidates if c in cols), None)

    if src_col is None or dst_col is None:
        raise ValueError(
            f"Could not detect source/target columns from: {list(df.columns)}"
        )

    if w_col is None:
        raise ValueError(
            f"Could not detect weight column from: {list(df.columns)}"
        )

    return src_col, dst_col, w_col


def load_inferred_edges(inferred_path: Path) -> tuple[list[tuple[str, str]], str, str, str]:
    """
    Load inferred edges from CSV and return edges ranked by descending weight.
    """

    if not inferred_path.exists():
        raise FileNotFoundError(f"Inferred file not found: {inferred_path}")

    df = pd.read_csv(inferred_path)
    src_col, dst_col, w_col = detect_columns(df)

    df = df[[src_col, dst_col, w_col]].copy()
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)
    df[w_col] = pd.to_numeric(df[w_col], errors="coerce")
    df = df.dropna(subset=[w_col])

    # Remove self loops just to keep evaluation cleaner
    df = df[df[src_col] != df[dst_col]]

    # Sort by descending inferred strength
    df = df.sort_values(w_col, ascending=False).reset_index(drop=True)

    inferred_edges = list(zip(df[src_col], df[dst_col]))
    return inferred_edges, src_col, dst_col, w_col


def to_undirected(edge: tuple[str, str]) -> tuple[str, str]:
    u, v = edge
    return tuple(sorted((u, v)))


def precision_at_k(pred_edges: list[tuple[str, str]], truth_edges: set[tuple[str, str]], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = pred_edges[:k]
    if not topk:
        return 0.0
    hits = sum(1 for e in topk if e in truth_edges)
    return hits / len(topk)


# =========================
# Main
# =========================

def main() -> None:
    root = project_root()
    inferred_path = INFERRED_PATH
    higgs_dir = HIGGS_DIR

    # Load inferred edges
    inferred_edges, src_col, dst_col, w_col = load_inferred_edges(inferred_path)

    # Load truth graph
    truth_edges = get_truth_edges(
        higgs_dir=higgs_dir,
        truth_type=TRUTH_TYPE,
        reverse=REVERSE_TRUTH,
    )

    # Undirected versions for relaxed evaluation
    truth_undirected = {to_undirected(e) for e in truth_edges}
    inferred_undirected = [to_undirected(e) for e in inferred_edges]

    # Evaluate precision@k
    K = min(TOP_K_MAX, len(inferred_edges))
    ks = list(range(1, K + 1))

    p_dir = [precision_at_k(inferred_edges, truth_edges, k) for k in ks]
    p_undir = [precision_at_k(inferred_undirected, truth_undirected, k) for k in ks]

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / f"higgs_{TRUTH_TYPE}_precision_at_k.csv"
    fig_path = out_dir / f"higgs_{TRUTH_TYPE}_precision_at_k.png"

    pd.DataFrame(
        {
            "k": ks,
            "precision_directed": p_dir,
            "precision_undirected": p_undir,
        }
    ).to_csv(metrics_csv, index=False)

    plt.figure()
    plt.plot(ks, p_dir, label="directed")
    plt.plot(ks, p_undir, label="undirected")
    plt.xlabel("k")
    plt.ylabel("precision@k")
    plt.title(f"Higgs: inferred edges vs {TRUTH_TYPE}")
    plt.legend()
    plt.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close()

    print("Inferred file:", inferred_path)
    print("Truth graph type:", TRUTH_TYPE)
    print("Reverse truth:", REVERSE_TRUTH)
    print("Detected inferred columns:", {"src": src_col, "dst": dst_col, "weight": w_col})
    print("Number of inferred edges:", len(inferred_edges))
    print("Number of truth edges:", len(truth_edges))
    print("Saved:", metrics_csv)
    print("Saved:", fig_path)

    print("\nQuick summary:")
    for k in [10, 50, 100, 500, 1000, 2000]:
        if k <= K:
            print(
                f"k={k:>4}  "
                f"precision_directed={p_dir[k-1]:.4f}  "
                f"precision_undirected={p_undir[k-1]:.4f}"
            )

    print("Unique inferred nodes:", len(set([u for e in inferred_edges for u in e])))
    print("Unique truth nodes:", len(set([u for e in truth_edges for u in e])))

    for k in [10, 50, 100, 500, 1000, 2000]:
        if k <= K:
            dir_hits = sum(1 for e in inferred_edges[:k] if e in truth_edges)
            undir_hits = sum(1 for e in inferred_undirected[:k] if e in truth_undirected)
            print(
                f"k={k:>4}  "
                f"hits_directed={dir_hits}  hits_undirected={undir_hits}  "
                f"precision_directed={p_dir[k - 1]:.4f}  precision_undirected={p_undir[k - 1]:.4f}"
            )

if __name__ == "__main__":
    main()