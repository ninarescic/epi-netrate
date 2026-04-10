from __future__ import annotations

from pathlib import Path
import sys
import gzip
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.higgs.ground_truth import get_truth_edges


# ----------------------------
# Configuration
# ----------------------------
INFERRED_PATH = ROOT / "outputs" / "higgs_rt_inferred_21_50_top50_10T.csv"
HIGGS_DIR = ROOT / "data" / "higgs"
OUT_PATH = ROOT / "outputs" / "higgs_network_collision_4way_activity_followers_10T.png"

REVERSE_TRUTH = True
TOP_K_INFERRED = 2000
SHOW_LABELS = False
LAYOUT_SEED = 7

# activity-based node size
ACTIVITY_ALLOWED_KINDS = {"RT"}
ACTIVITY_MODE = "actor"   # "actor" or "any"


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r", encoding="utf-8")


def find_activity_file(higgs_dir: Path) -> Path:
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


def find_social_file(higgs_dir: Path) -> Path:
    candidates = [
        higgs_dir / "higgs-social_network.edgelist.gz",
        higgs_dir / "social_network.edgelist.gz",
        higgs_dir / "higgs-social_network.edgelist",
        higgs_dir / "social_network.edgelist",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find social network file in {higgs_dir}")


def load_activity_counts(
    higgs_dir: Path,
    *,
    allowed_kinds: set[str] | None,
    mode: str = "actor",
) -> dict[str, int]:
    path = find_activity_file(higgs_dir)
    counts = Counter()

    with _open_maybe_gzip(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue

            a, b, ts, kind = parts
            if allowed_kinds is not None and kind not in allowed_kinds:
                continue

            a = str(a)
            b = str(b)

            if mode == "actor":
                counts[a] += 1
            elif mode == "any":
                counts[a] += 1
                if b != a:
                    counts[b] += 1
            else:
                raise ValueError("mode must be 'actor' or 'any'")

    return dict(counts)


def load_follower_counts(higgs_dir: Path) -> dict[str, int]:
    """
    Count followers from the raw social graph.
    If the file is u v and means u follows v, then v gets +1 follower.
    """
    path = find_social_file(higgs_dir)
    counts = Counter()

    with _open_maybe_gzip(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            u, v = str(parts[0]), str(parts[1])
            if u != v:
                counts[v] += 1

    return dict(counts)


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

    activity_counts = load_activity_counts(
        HIGGS_DIR,
        allowed_kinds=ACTIVITY_ALLOWED_KINDS,
        mode=ACTIVITY_MODE,
    )
    follower_counts = load_follower_counts(HIGGS_DIR)

    truth_1hop_edges = sorted(truth_1hop)
    truth_2hop_only_edges = sorted(truth_2hop_only)

    G_draw = nx.DiGraph()
    G_draw.add_nodes_from(inferred_nodes)
    G_draw.add_edges_from(truth_1hop_edges)
    G_draw.add_edges_from(truth_2hop_only_edges)
    G_draw.add_edges_from(inferred_edges)

    G_layout = nx.Graph()
    G_layout.add_nodes_from(inferred_nodes)
    G_layout.add_edges_from(inferred_edges)

    pos = component_layout_packed(G_layout, seed=LAYOUT_SEED)

    node_sizes = [
        120 + 60 * np.log1p(activity_counts.get(node, 0))
        for node in G_draw.nodes()
    ]

    follower_vals = np.array([follower_counts.get(node, 0) for node in G_draw.nodes()], dtype=float)
    norm = mcolors.LogNorm(
        vmin=max(1.0, follower_vals[follower_vals > 0].min() if np.any(follower_vals > 0) else 1.0),
        vmax=max(1.0, follower_vals.max() if len(follower_vals) else 1.0),
    )
    cmap = plt.cm.Blues
    node_colors = cmap(norm(np.maximum(follower_vals, 1.0)))

    plt.figure(figsize=(25, 20))

    nx.draw_networkx_nodes(
        G_draw,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.7,
    )

    if SHOW_LABELS:
        nx.draw_networkx_labels(G_draw, pos, font_size=7)

    draw_directed_edge_group(
        G_draw, pos, truth_2hop_only_edges,
        color="lightgray", style="dashed", width=0.9, alpha=0.18, arrowsize=9
    )
    draw_directed_edge_group(
        G_draw, pos, truth_1hop_edges,
        color="gray", style="solid", width=1.0, alpha=0.22, arrowsize=9
    )
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

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.7, pad=0.02)
    cbar.set_label("number of followers (log scale)")

    plt.title(
        f"Higgs inferred-edge collisions, 4-way\n"
        f"node size = log activity, node color = followers, top {len(inferred_edges)} inferred edges"
    )
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
    print("Min plotted followers:", int(follower_vals.min()) if len(follower_vals) else 0)
    print("Max plotted followers:", int(follower_vals.max()) if len(follower_vals) else 0)


if __name__ == "__main__":
    main()