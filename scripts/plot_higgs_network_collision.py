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
OUT_PATH = ROOT / "outputs" / "higgs_network_collision_4way.png"

REVERSE_TRUTH = True
TOP_K_INFERRED = 80
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
            G, pos,
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
            G, pos,
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
            G, pos,
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


def component_layout_packed(G: nx.Graph, seed: int = 7) -> dict[str, tuple[float, float]]:
    pos_all: dict[str, tuple[float, float]] = {}
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    x_offset = 0.0
    y_offset = 0.0
    row_height = 0.0
    max_row_width = 12.0
    gap = 2.5

    for comp_nodes in components:
        H = G.subgraph(comp_nodes).copy()
        n = H.number_of_nodes()

        if n == 1:
            node = next(iter(H.nodes()))
            local_pos = {node: (0.0, 0.0)}
            width = 1.0
            height = 1.0
        else:
            local_pos = nx.spring_layout(
                H,
                seed=seed,
                k=2.6 / (n ** 0.5),
                iterations=300,
            )

            xs = [p[0] for p in local_pos.values()]
            ys = [p[1] for p in local_pos.values()]
            width = max(xs) - min(xs) if xs else 1.0
            height = max(ys) - min(ys) if ys else 1.0

            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            local_pos = {u: (x - cx, y - cy) for u, (x, y) in local_pos.items()}

        if x_offset + width > max_row_width:
            x_offset = 0.0
            y_offset -= (row_height + gap)
            row_height = 0.0

        for u, (x, y) in local_pos.items():
            pos_all[u] = (x + x_offset, y + y_offset)

        x_offset += width + gap
        row_height = max(row_height, height)

    return pos_all


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
    # 3) Classify inferred edges
    # ----------------------------
    hit_1hop = []
    hit_2hop_only = []
    reverse_only = []
    miss = []

    for e in inferred_edges:
        u, v = e
        rev = (v, u)

        if e in truth_1hop:
            hit_1hop.append(e)
        elif e in truth_2hop_only:
            hit_2hop_only.append(e)
        elif rev in truth_2hop:
            reverse_only.append(e)
        else:
            miss.append(e)

    # ----------------------------
    # 4) Background truth for context
    # ----------------------------
    truth_1hop_edges = sorted(truth_1hop)
    truth_2hop_only_edges = sorted(truth_2hop_only)

    # ----------------------------
    # 5) Graph for drawing and layout
    # ----------------------------
    G_draw = nx.DiGraph()
    G_draw.add_nodes_from(inferred_nodes)
    G_draw.add_edges_from(truth_1hop_edges)
    G_draw.add_edges_from(truth_2hop_only_edges)
    G_draw.add_edges_from(inferred_edges)

    G_layout = nx.Graph()
    G_layout.add_nodes_from(inferred_nodes)
    G_layout.add_edges_from(inferred_edges)

    pos = component_layout_packed(G_layout, seed=LAYOUT_SEED)

    # ----------------------------
    # 6) Draw
    # ----------------------------
    plt.figure(figsize=(16, 12))

    nx.draw_networkx_nodes(
        G_draw,
        pos,
        node_size=260,
        edgecolors="black",
        linewidths=0.7,
    )

    if SHOW_LABELS:
        nx.draw_networkx_labels(G_draw, pos, font_size=7)

    # light truth background
    draw_directed_edge_group(
        G_draw, pos, truth_2hop_only_edges,
        color="lightgray", style="dashed", width=0.9, alpha=0.18, arrowsize=9
    )
    draw_directed_edge_group(
        G_draw, pos, truth_1hop_edges,
        color="gray", style="solid", width=1.0, alpha=0.22, arrowsize=9
    )

    # inferred edges by class
    draw_directed_edge_group(
        G_draw, pos, hit_1hop,
        color="darkgreen", style="solid", width=2.5, alpha=0.95, arrowsize=16
    )
    draw_directed_edge_group(
        G_draw, pos, hit_2hop_only,
        color="limegreen", style="solid", width=2.4, alpha=0.92, arrowsize=16
    )
    draw_directed_edge_group(
        G_draw, pos, reverse_only,
        color="orange", style="solid", width=2.2, alpha=0.90, arrowsize=16
    )
    draw_directed_edge_group(
        G_draw, pos, miss,
        color="red", style="solid", width=1.9, alpha=0.78, arrowsize=15
    )

    legend_items = [
        Line2D([0], [0], color="darkgreen", lw=2, linestyle="solid", label="directed hit: 1-hop"),
        Line2D([0], [0], color="limegreen", lw=2, linestyle="solid", label="directed hit: 2-hop only"),
        Line2D([0], [0], color="orange", lw=2, linestyle="solid", label="reverse-direction only"),
        Line2D([0], [0], color="red", lw=2, linestyle="solid", label="neither direction"),
        Line2D([0], [0], color="black", lw=2, linestyle="solid", label="background: 1-hop"),
        Line2D([0], [0], color="lightgray", lw=2, linestyle="dashed", label="background: 2-hop only"),
    ]
    plt.legend(handles=legend_items, loc="upper right")

    plt.title(f"Higgs inferred-edge collisions, 4-way (top {len(inferred_edges)} inferred edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close()

    print("Saved:", OUT_PATH)
    print("Nodes plotted:", len(inferred_nodes))
    print("Inferred edges plotted:", len(inferred_edges))
    print("Directed 1-hop hits:", len(hit_1hop))
    print("Directed 2-hop-only hits:", len(hit_2hop_only))
    print("Reverse-direction-only hits:", len(reverse_only))
    print("Neither-direction misses:", len(miss))
    print("Background 1-hop truth edges:", len(truth_1hop_edges))
    print("Background 2-hop-only truth edges:", len(truth_2hop_only_edges))


if __name__ == "__main__":
    main()