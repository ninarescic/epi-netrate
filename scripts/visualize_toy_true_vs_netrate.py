from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List, Set

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from epinet.paths import project_root


# --- EDIT THIS: define your toy ground-truth directed edges here ---
TRUE_EDGES: Set[Tuple[str, str]] = {
    ("1", "2"),
    ("1", "3"),
    ("2", "1"),
    ("2", "3"),
}
# ---------------------------------------------------------------


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def detect_schema(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """
    Try to infer which columns are: source, target, weight.
    Returns (src_col, dst_col, weight_col|None).
    """
    cols = list(df.columns)

    # Common patterns
    src = _pick_col(cols, ["src", "source", "u", "from", "i", "node_u", "node_i"])
    dst = _pick_col(cols, ["dst", "target", "v", "to", "j", "node_v", "node_j"])

    # If not found, try first two columns
    if src is None or dst is None:
        if len(cols) < 2:
            raise ValueError(f"Need at least 2 columns for edges, got: {cols}")
        src = src or cols[0]
        dst = dst or cols[1]

    w = _pick_col(cols, ["weight", "rate", "alpha", "score", "w", "value", "lambda"])
    return src, dst, w


def precision_at_k(inferred_edges: List[Tuple[str, str]], true_edges: Set[Tuple[str, str]], k: int) -> float:
    topk = inferred_edges[:k]
    if not topk:
        return 0.0
    hits = sum(1 for e in topk if e in true_edges)
    return hits / len(topk)


def main():
    root = project_root()
    inferred_path = root / "outputs" / "toy_inferred.csv"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inferred_path.exists():
        raise FileNotFoundError(f"Missing {inferred_path}. Run toy baseline inference first.")

    df = pd.read_csv(inferred_path)
    if df.empty:
        raise ValueError(f"{inferred_path} is empty.")

    src_col, dst_col, w_col = detect_schema(df)

    # Normalize to strings (your legacy code uses str node_ids)
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)

    # Sort by weight if available, else keep file order
    if w_col is not None:
        df[w_col] = pd.to_numeric(df[w_col], errors="coerce")
        df = df.sort_values(by=w_col, ascending=False, na_position="last")

    inferred_edges = list(zip(df[src_col].tolist(), df[dst_col].tolist()))

    # ---- Precision@k curve ----
    K = min(50, len(inferred_edges))
    ks = list(range(1, K + 1))
    ps = [precision_at_k(inferred_edges, TRUE_EDGES, k) for k in ks]

    plt.figure()
    plt.plot(ks, ps)
    plt.xlabel("k")
    plt.ylabel("precision@k")
    plt.title("Toy: precision@k (true vs inferred)")
    pfig = out_dir / "toy_precision_at_k.png"
    plt.savefig(pfig, bbox_inches="tight", dpi=200)
    plt.close()

    print("Detected columns:")
    print(f"  src: {src_col}")
    print(f"  dst: {dst_col}")
    print(f"  weight: {w_col}")
    print("\nPrecision@k:")
    for k, p in zip(ks, ps):
        if k in {1, 2, 3, 5, 10, 20, 50}:
            print(f"  k={k:>2}: {p:.3f}")
    print(f"\nSaved: {pfig}")

    # ---- Graph overlay visualization ----
    # Build graphs
    G_true = nx.DiGraph()
    G_true.add_edges_from(TRUE_EDGES)

    # Choose top N inferred edges for drawing
    N = min(20, len(inferred_edges))
    topN = inferred_edges[:N]
    G_inf = nx.DiGraph()
    G_inf.add_edges_from(topN)

    # Union for layout so nodes align
    G_all = nx.DiGraph()
    G_all.add_nodes_from(set(G_true.nodes()) | set(G_inf.nodes()))
    G_all.add_edges_from(G_true.edges())
    G_all.add_edges_from(G_inf.edges())

    # Layout
    pos = nx.spring_layout(G_all, seed=7)

    # Edge categories
    inf_set = set(topN)
    true_set = set(TRUE_EDGES)
    overlap = list(inf_set & true_set)
    only_inf = list(inf_set - true_set)
    only_true = list(true_set - inf_set)

    plt.figure()
    nx.draw_networkx_nodes(G_all, pos)
    nx.draw_networkx_labels(G_all, pos, font_size=10)

    # Draw edges without setting explicit colors (matplotlib defaults vary),
    # but we DO need distinction. We'll use different styles instead of colors.
    nx.draw_networkx_edges(G_all, pos, edgelist=only_true, style="dashed", arrows=True)
    nx.draw_networkx_edges(G_all, pos, edgelist=only_inf, style="dotted", arrows=True)
    nx.draw_networkx_edges(G_all, pos, edgelist=overlap, style="solid", arrows=True, width=2.5)

    plt.title(f"Toy overlay (top {N} inferred): solid=overlap, dashed=true-only, dotted=inferred-only")
    gfig = out_dir / "toy_graph_overlay.png"
    plt.savefig(gfig, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"Saved: {gfig}")


if __name__ == "__main__":
    main()
