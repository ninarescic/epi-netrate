from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict

import os
import pandas as pd
import networkx as nx

# Headless server safe
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
    Robust to extra columns (weights): only first two tokens are used.
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


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def detect_inferred_schema(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    cols = list(df.columns)
    src = _pick_col(cols, ["src", "source", "u", "from", "i", "node_u", "node_i"])
    dst = _pick_col(cols, ["dst", "target", "v", "to", "j", "node_v", "node_j"])
    if src is None or dst is None:
        if len(cols) < 2:
            raise ValueError(f"Need at least 2 columns for edges, got: {cols}")
        src = src or cols[0]
        dst = dst or cols[1]
    w = _pick_col(cols, ["weight", "rate", "alpha", "score", "w", "value", "lambda"])
    return src, dst, w


def main():
    root = project_root()
    higgs_dir = get_higgs_dir()

    inferred_path = root / "outputs" / "higgs_rt_subset_inferred.csv"
    if not inferred_path.exists():
        raise FileNotFoundError(f"Missing inferred file: {inferred_path}")

    # ---------- knobs ----------
    TOP_INFERRED_EDGES = 2000   # increase later (5000) if you like
    TOP_NODES = 200             # 100–300 is usually readable
    # --------------------------

    df = pd.read_csv(inferred_path)
    if df.empty:
        raise ValueError(f"{inferred_path} is empty.")

    src_col, dst_col, w_col = detect_inferred_schema(df)
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)

    if w_col is not None:
        df[w_col] = pd.to_numeric(df[w_col], errors="coerce")
        df = df.sort_values(by=w_col, ascending=False, na_position="last")

    # Top inferred edges
    df_top = df.head(min(TOP_INFERRED_EDGES, len(df))).copy()
    inferred_edges = list(zip(df_top[src_col].tolist(), df_top[dst_col].tolist()))

    # Choose top nodes by inferred degree (in+out)
    deg: Dict[str, int] = {}
    for u, v in inferred_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    top_nodes = [n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:TOP_NODES]]
    top_nodes_set = set(top_nodes)

    # Induce inferred subgraph on those nodes
    inferred_edges_sub = [(u, v) for (u, v) in inferred_edges if u in top_nodes_set and v in top_nodes_set]
    inferred_set = set(inferred_edges_sub)

    # Load proxy and restrict to those nodes
    proxy_edges = load_rt_proxy_edges(higgs_dir)
    proxy_edges_sub = [(u, v) for (u, v) in proxy_edges if u in top_nodes_set and v in top_nodes_set]
    proxy_set = set(proxy_edges_sub)

    overlap = list(inferred_set & proxy_set)
    inf_only = list(inferred_set - proxy_set)
    proxy_only = list(proxy_set - inferred_set)

    # Build union graph for layout
    G = nx.DiGraph()
    G.add_nodes_from(top_nodes_set)
    G.add_edges_from(inferred_edges_sub)
    G.add_edges_from(proxy_edges_sub)

    if G.number_of_edges() == 0:
        raise RuntimeError("No edges in the plotted subgraph. Try increasing TOP_INFERRED_EDGES or TOP_NODES.")

    # Layout (can be slow if graph is dense; keep TOP_NODES reasonable)
    pos = nx.spring_layout(G, seed=7)

    # Draw
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=80)
    nx.draw_networkx_labels(G, pos, font_size=6)

    # Draw edges with colors
    nx.draw_networkx_edges(
        G, pos,
        edgelist=proxy_only,
        edge_color="lightgray",
        arrows=False,
        alpha=0.6,
        width=1.0,
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=inf_only,
        edge_color="red",
        arrows=False,
        alpha=0.6,
        width=1.2,
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=overlap,
        edge_color="green",
        arrows=False,
        alpha=0.9,
        width=2.5,
    )

    plt.title(
        f"Higgs RT overlay (nodes={len(top_nodes_set)} | inferred edges={len(inferred_edges_sub)} | "
        f"proxy edges={len(proxy_edges_sub)} | overlap={len(overlap)})\n"
        "solid=overlap, dashed=proxy-only, dotted=inferred-only"
    )

    out_path = root / "outputs" / "higgs_rt_overlay_graph.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()

    print("Saved:", out_path)
    print("Inferred:", inferred_path)
    print("Proxy RT:", higgs_dir / "higgs-retweet_network.edgelist")
    print("Detected inferred columns:", {"src": src_col, "dst": dst_col, "weight": w_col})
    print("Counts:", {"nodes": len(top_nodes_set), "inf_edges": len(inferred_edges_sub), "proxy_edges": len(proxy_edges_sub), "overlap": len(overlap)})


if __name__ == "__main__":
    main()
