from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.higgs.ground_truth import get_truth_edges


# ----------------------------
# Configuration
# ----------------------------
INFERRED_PATH = ROOT / "outputs" / "higgs_rt_inferred_11_20_top20.csv"
HIGGS_DIR = ROOT / "data" / "higgs"
OUT_PATH = ROOT / "outputs" / "higgs_network_collisions_top100.png"

REVERSE_TRUTH = True
TOP_K_INFERRED = 60
SHOW_LABELS = False
LAYOUT_SEED = 7


def detect_inferred_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = {c.lower(): c for c in df.columns}

    src = cols.get("source") or cols.get("src")
    dst = cols.get("target") or cols.get("dst")
    weight = cols.get("beta") or cols.get("weight")

    if src is None or dst is None or weight is None:
        raise ValueError(f"Could not detect source/target/weight columns in: {list(df.columns)}")

    return src, dst, weight


def split_edges_for_curved_arrows(edges: list[tuple[str, str]]):
    edge_set = set(edges)
    single = []
    mutual_pos = []
    mutual_neg = []
    seen_pairs = set()

    for u, v in edges:
        if (v, u) in edge_set:
            pair = tuple(sorted((u, v)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            a, b = pair
            if (a, b) in edge_set:
                mutual_pos.append((a, b))
            if (b, a) in edge_set:
                mutual_neg.append((b, a))
        else:
            single.append((u, v))

    return single, mutual_pos, mutual_neg


def draw_directed_edge_group(
    G: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    *,
    color: str,
    style: str,
    width: float,
    alpha: float,
    arrowsize: int,
) -> None:
    if not edges:
        return

    single, mutual_pos, mutual_neg = split_edges_for_curved_arrows(edges)

    if single:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=single,
            edge_color=color,
            style=style,
            width=width,
            alpha=alpha,
            arrows=True,
            arrowsize=arrowsize,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.0",
        )

    if mutual_pos:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=mutual_pos,
            edge_color=color,
            style=style,
            width=width,
            alpha=alpha,
            arrows=True,
            arrowsize=arrowsize,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.18",
        )

    if mutual_neg:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=mutual_neg,
            edge_color=color,
            style=style,
            width=width,
            alpha=alpha,
            arrows=True,
            arrowsize=arrowsize,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.18",
        )


def main() -> None:
    # ----------------------------
    # 1) Load inferred edges
    # ----------------------------
    inferred_df = pd.read_csv(INFERRED_PATH)
    src_col, dst_col, w_col = detect_inferred_columns(inferred_df)

    inferred_df = inferred_df[[src_col, dst_col, w_col]].copy()
    inferred_df.columns = ["source", "target", "weight"]

    inferred_df["source"] = inferred_df["source"].astype(str)
    inferred_df["target"] = inferred_df["target"].astype(str)
    inferred_df["weight"] = pd.to_numeric(inferred_df["weight"], errors="coerce")

    inferred_df = inferred_df.dropna(subset=["weight"])
    inferred_df = inferred_df[inferred_df["source"] != inferred_df["target"]]
    inferred_df = inferred_df.sort_values("weight", ascending=False)

    if TOP_K_INFERRED is not None:
        inferred_df = inferred_df.head(TOP_K_INFERRED).copy()

    inferred_edges = list(zip(inferred_df["source"], inferred_df["target"]))
    inferred_nodes = sorted(set(inferred_df["source"]).union(set(inferred_df["target"])))
    node_set = set(inferred_nodes)

    # ----------------------------
    # 2) Load truth edges
    # ----------------------------
    truth_1hop_raw = get_truth_edges(
        higgs_dir=HIGGS_DIR,
        truth_type="social_1hop",
        reverse=REVERSE_TRUTH,
    )

    truth_2hop_raw = get_truth_edges(
        higgs_dir=HIGGS_DIR,
        truth_type="social_2hop",
        reverse=REVERSE_TRUTH,
    )

    truth_1hop = {
        (str(u), str(v))
        for (u, v) in truth_1hop_raw
        if str(u) in node_set and str(v) in node_set and str(u) != str(v)
    }

    truth_2hop = {
        (str(u), str(v))
        for (u, v) in truth_2hop_raw
        if str(u) in node_set and str(v) in node_set and str(u) != str(v)
    }

    truth_2hop_only = truth_2hop - truth_1hop

    # ----------------------------
    # 3) Split inferred edges by collision
    # ----------------------------
    inferred_hit_edges = [e for e in inferred_edges if e in truth_2hop]
    inferred_miss_edges = [e for e in inferred_edges if e not in truth_2hop]

    # optional light gray background truth
    truth_1hop_edges = sorted(truth_1hop)
    truth_2hop_only_edges = sorted(truth_2hop_only)

    # ----------------------------
    # 4) Build graph for layout
    # ----------------------------
    G = nx.DiGraph()
    G.add_nodes_from(inferred_nodes)
    G.add_edges_from(truth_1hop_edges)
    G.add_edges_from(truth_2hop_only_edges)
    G.add_edges_from(inferred_edges)

    pos = nx.spring_layout(G.to_undirected(), seed=LAYOUT_SEED)

    # ----------------------------
    # 5) Draw
    # ----------------------------
    plt.figure(figsize=(14, 12))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=320,
        edgecolors="black",
        linewidths=0.8,
    )

    if SHOW_LABELS:
        nx.draw_networkx_labels(G, pos, font_size=8)

    # background truth
    draw_directed_edge_group(
        G,
        pos,
        truth_2hop_only_edges,
        color="lightgray",
        style="dashed",
        width=1.0,
        alpha=0.25,
        arrowsize=10,
    )

    draw_directed_edge_group(
        G,
        pos,
        truth_1hop_edges,
        color="gray",
        style="solid",
        width=1.2,
        alpha=0.30,
        arrowsize=10,
    )

    # inferred edges that collide with truth = green
    draw_directed_edge_group(
        G,
        pos,
        inferred_hit_edges,
        color="green",
        style="solid",
        width=2.3,
        alpha=0.9,
        arrowsize=16,
    )

    # inferred edges that do not collide = red
    draw_directed_edge_group(
        G,
        pos,
        inferred_miss_edges,
        color="red",
        style="solid",
        width=2.0,
        alpha=0.8,
        arrowsize=16,
    )

    legend_items = [
        Line2D([0], [0], color="gray", lw=2, linestyle="solid", label="social_1hop"),
        Line2D([0], [0], color="lightgray", lw=2, linestyle="dashed", label="social_2hop only"),
        Line2D([0], [0], color="green", lw=2, linestyle="solid", label="inferred ∩ proxy truth"),
        Line2D([0], [0], color="red", lw=2, linestyle="solid", label="inferred only"),
    ]
    plt.legend(handles=legend_items, loc="upper right")

    plt.title(f"Higgs inferred-edge collisions (top {len(inferred_edges)} inferred edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close()

    print("Saved:", OUT_PATH)
    print("Nodes plotted:", len(inferred_nodes))
    print("Inferred edges plotted:", len(inferred_edges))
    print("Green inferred edges (collide with truth):", len(inferred_hit_edges))
    print("Red inferred edges (not in truth):", len(inferred_miss_edges))
    print("Background 1-hop truth edges:", len(truth_1hop_edges))
    print("Background 2-hop-only truth edges:", len(truth_2hop_only_edges))


if __name__ == "__main__":
    main()