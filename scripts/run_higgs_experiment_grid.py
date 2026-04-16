from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[0]


@dataclass
class ExperimentConfig:
    method: str
    label: str
    cascades_path: Path
    out_path: Path
    params: dict[str, Any]


TRUTH_TYPES = ("social_1hop", "social_2hop")
REVERSE_TRUTH = True
TOP_K_SUMMARY = (10, 50, 100, 500, 1000, 2000)

TOP_K_OVERLAP_DEFAULT = (50, 100, 500)


def project_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "epinet").exists() and (cwd / "scripts").exists():
        return cwd
    return ROOT


def ensure_repo_on_path(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def detect_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = {c.lower(): c for c in df.columns}
    src_candidates = ["source", "src", "u", "parent", "from"]
    dst_candidates = ["target", "dst", "v", "child", "to"]
    weight_candidates = ["beta", "weight", "score", "alpha", "value"]

    src_col = next((cols[c] for c in src_candidates if c in cols), None)
    dst_col = next((cols[c] for c in dst_candidates if c in cols), None)
    w_col = next((cols[c] for c in weight_candidates if c in cols), None)

    if src_col is None or dst_col is None or w_col is None:
        raise ValueError(f"Could not detect source/target/weight columns from: {list(df.columns)}")

    return src_col, dst_col, w_col


def load_inferred_edges(inferred_path: Path) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    if not inferred_path.exists():
        raise FileNotFoundError(f"Inferred file not found: {inferred_path}")

    df = pd.read_csv(inferred_path)
    src_col, dst_col, w_col = detect_columns(df)
    work = df[[src_col, dst_col, w_col]].copy()
    work[src_col] = work[src_col].astype(str)
    work[dst_col] = work[dst_col].astype(str)
    work[w_col] = pd.to_numeric(work[w_col], errors="coerce")
    work = work.dropna(subset=[w_col])
    work = work[work[src_col] != work[dst_col]].copy()
    work = work.sort_values(w_col, ascending=False).reset_index(drop=True)
    edges = list(zip(work[src_col], work[dst_col]))
    return work.rename(columns={src_col: "source", dst_col: "target", w_col: "weight"}), edges


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


def prepare_relative_cascades(cascades_path: Path, force: bool = False) -> Path:
    relative_path = cascades_path.with_name(f"{cascades_path.stem}_relative.csv")
    if relative_path.exists() and not force:
        return relative_path

    df = pd.read_csv(cascades_path)
    required = {"cascade_id", "node_id", "infection_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cascade file missing required columns: {sorted(missing)}")

    df = df.copy()
    df["infection_time"] = pd.to_numeric(df["infection_time"], errors="raise")
    df["infection_time"] = (
        df["infection_time"] - df.groupby("cascade_id")["infection_time"].transform("min")
    )
    df.to_csv(relative_path, index=False)
    return relative_path


def build_experiment_grid(
    cascades_path: Path,
    output_dir: Path,
    methods: Iterable[str],
    diffusion_types: Iterable[str],
    horizons: Iterable[float],
    l1: float,
    thr: float,
    solver: str | None,
    matlab_faithful: bool,
) -> list[ExperimentConfig]:
    experiments: list[ExperimentConfig] = []
    for method in methods:
        for diffusion_type in diffusion_types:
            for horizon in horizons:
                label = f"{method}_{diffusion_type}_{format_horizon_tag(horizon)}"
                out_path = output_dir / label / "inferred_edges.csv"
                params: dict[str, Any] = {
                    "l1": l1,
                    "thr": thr,
                    "type_diffusion": diffusion_type,
                    "horizon": float(horizon),
                }
                if solver:
                    params["solver"] = solver
                if matlab_faithful:
                    params["matlab_faithful"] = True
                experiments.append(
                    ExperimentConfig(
                        method=method,
                        label=label,
                        cascades_path=cascades_path,
                        out_path=out_path,
                        params=params,
                    )
                )
    return experiments


def format_horizon_tag(horizon: float) -> str:
    if horizon <= 0:
        return "h0"
    exponent = int(math.floor(math.log10(abs(horizon)))) if horizon != 0 else 0
    mantissa = horizon / (10 ** exponent)
    if abs(mantissa - round(mantissa)) < 1e-12:
        mantissa_str = str(int(round(mantissa)))
    else:
        mantissa_str = f"{mantissa:.3g}".replace(".", "p")
    return f"h{mantissa_str}e{exponent}"


def run_inference(method: str, cascades_path: Path, out_path: Path, params: dict[str, Any], repo_root: Path) -> None:
    ensure_repo_on_path(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "netrate_baseline":
        from epinet.baseline import infer_netrate_baseline

        infer_netrate_baseline(cascades_path=cascades_path, out_path=out_path, params=params)
        return

    if method == "epi_netrate":
        raise NotImplementedError("epi_netrate is not wired yet. Add your future function call here.")

    raise ValueError(f"Unsupported method: {method}")


def evaluate_one_truth(
    inferred_path: Path,
    truth_type: str,
    higgs_dir: Path,
    reverse_truth: bool,
    top_k_max: int,
    repo_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_repo_on_path(repo_root)
    from datasets.higgs.ground_truth import get_truth_edges

    inferred_df, inferred_edges = load_inferred_edges(inferred_path)
    truth_edges = get_truth_edges(higgs_dir=higgs_dir, truth_type=truth_type, reverse=reverse_truth)

    inferred_undirected = [to_undirected(e) for e in inferred_edges]
    truth_undirected = {to_undirected(e) for e in truth_edges}

    K = min(top_k_max, len(inferred_edges))
    ks = list(range(1, K + 1))
    p_dir = [precision_at_k(inferred_edges, truth_edges, k) for k in ks]
    p_undir = [precision_at_k(inferred_undirected, truth_undirected, k) for k in ks]
    hits_dir = [sum(1 for e in inferred_edges[:k] if e in truth_edges) for k in ks]
    hits_undir = [sum(1 for e in inferred_undirected[:k] if e in truth_undirected) for k in ks]

    metrics_df = pd.DataFrame(
        {
            "k": ks,
            "precision_directed": p_dir,
            "precision_undirected": p_undir,
            "hits_directed": hits_dir,
            "hits_undirected": hits_undir,
        }
    )

    summary: dict[str, Any] = {
        "truth_type": truth_type,
        "reverse_truth": reverse_truth,
        "num_inferred_edges": len(inferred_edges),
        "num_truth_edges": len(truth_edges),
        "num_inferred_nodes": len(set([u for e in inferred_edges for u in e])),
        "num_truth_nodes": len(set([u for e in truth_edges for u in e])),
        "top_edges_json": inferred_df.head(10).to_dict(orient="records"),
    }
    for k in TOP_K_SUMMARY:
        if k <= K:
            row = metrics_df.iloc[k - 1]
            summary[f"p_at_{k}_directed"] = float(row["precision_directed"])
            summary[f"p_at_{k}_undirected"] = float(row["precision_undirected"])
            summary[f"hits_at_{k}_directed"] = int(row["hits_directed"])
            summary[f"hits_at_{k}_undirected"] = int(row["hits_undirected"])
        else:
            summary[f"p_at_{k}_directed"] = None
            summary[f"p_at_{k}_undirected"] = None
            summary[f"hits_at_{k}_directed"] = None
            summary[f"hits_at_{k}_undirected"] = None

    return metrics_df, summary


def save_precision_plot(metrics_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure()
    plt.plot(metrics_df["k"], metrics_df["precision_directed"], label="directed")
    plt.plot(metrics_df["k"], metrics_df["precision_undirected"], label="undirected")
    plt.xlabel("k")
    plt.ylabel("precision@k")
    plt.title(title)
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def save_comparison_plots(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if summary_df.empty:
        return saved

    for truth_type in sorted(summary_df["truth_type"].dropna().unique()):
        sub = summary_df[summary_df["truth_type"] == truth_type].copy()
        if sub.empty:
            continue

        if "p_at_100_directed" in sub.columns and sub["p_at_100_directed"].notna().any():
            plt.figure()
            x = range(len(sub))
            plt.bar(x, sub["p_at_100_directed"].fillna(0.0))
            plt.xticks(list(x), sub["label"], rotation=45, ha="right")
            plt.ylabel("precision@100 (directed)")
            plt.title(f"Comparison across runs: {truth_type}")
            out_path = output_dir / f"summary_{truth_type}_p_at_100.png"
            plt.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close()
            saved.append(out_path)

        plt.figure()
        x = range(len(sub))
        plt.bar(x, sub["num_inferred_edges"].fillna(0))
        plt.xticks(list(x), sub["label"], rotation=45, ha="right")
        plt.ylabel("number of inferred edges")
        plt.title(f"Inferred edge count by run: {truth_type}")
        out_path = output_dir / f"summary_{truth_type}_edge_counts.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
        saved.append(out_path)

    return saved


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
    matches = list(higgs_dir.glob("*activity_time*"))
    if matches:
        return matches[0]
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


def load_activity_counts(higgs_dir: Path, *, allowed_kinds: set[str] | None, mode: str = "actor") -> dict[str, int]:
    path = find_activity_file(higgs_dir)
    counts: Counter[str] = Counter()
    with _open_maybe_gzip(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            a, b, _ts, kind = parts
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
    path = find_social_file(higgs_dir)
    counts: Counter[str] = Counter()
    with _open_maybe_gzip(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = str(parts[0]), str(parts[1])
            if u != v:
                counts[v] += 1
    return dict(counts)


def split_edges_for_curved_arrows(edges: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    edge_set = set(edges)
    single: list[tuple[str, str]] = []
    mutual_pos: list[tuple[str, str]] = []
    mutual_neg: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
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
            G, pos, edgelist=single, edge_color=color, style=style, width=width, alpha=alpha,
            arrows=True, arrowsize=arrowsize, arrowstyle="-|>", connectionstyle="arc3,rad=0.0"
        )
    if mutual_pos:
        nx.draw_networkx_edges(
            G, pos, edgelist=mutual_pos, edge_color=color, style=style, width=width, alpha=alpha,
            arrows=True, arrowsize=arrowsize, arrowstyle="-|>", connectionstyle="arc3,rad=0.18"
        )
    if mutual_neg:
        nx.draw_networkx_edges(
            G, pos, edgelist=mutual_neg, edge_color=color, style=style, width=width, alpha=alpha,
            arrows=True, arrowsize=arrowsize, arrowstyle="-|>", connectionstyle="arc3,rad=-0.18"
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
            local_pos = nx.spring_layout(H, seed=seed, k=2.6 / (n ** 0.5), iterations=300)
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


def save_collision_activity_plot(
    inferred_path: Path,
    out_path: Path,
    higgs_dir: Path,
    *,
    repo_root: Path,
    reverse_truth: bool = True,
    top_k_inferred: int = 2000,
    show_labels: bool = False,
    layout_seed: int = 7,
    activity_allowed_kinds: set[str] | None = None,
    activity_mode: str = "actor",
) -> dict[str, Any]:
    ensure_repo_on_path(repo_root)
    from datasets.higgs.ground_truth import get_truth_edges

    inferred_df, inferred_edges = load_inferred_edges(inferred_path)
    if top_k_inferred is not None:
        inferred_df = inferred_df.head(top_k_inferred).copy()
        inferred_edges = list(zip(inferred_df["source"], inferred_df["target"]))

    inferred_nodes = sorted(set(inferred_df["source"]).union(set(inferred_df["target"])))
    node_set = set(inferred_nodes)

    truth_1hop_raw = get_truth_edges(higgs_dir=higgs_dir, truth_type="social_1hop", reverse=reverse_truth)
    truth_2hop_raw = get_truth_edges(higgs_dir=higgs_dir, truth_type="social_2hop", reverse=reverse_truth)
    truth_1hop = {(str(u), str(v)) for (u, v) in truth_1hop_raw if str(u) in node_set and str(v) in node_set and str(u) != str(v)}
    truth_2hop = {(str(u), str(v)) for (u, v) in truth_2hop_raw if str(u) in node_set and str(v) in node_set and str(u) != str(v)}
    truth_2hop_only = truth_2hop - truth_1hop

    hit_1hop: list[tuple[str, str]] = []
    hit_2hop_only: list[tuple[str, str]] = []
    reverse_only: list[tuple[str, str]] = []
    miss: list[tuple[str, str]] = []
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
        higgs_dir,
        allowed_kinds=activity_allowed_kinds if activity_allowed_kinds is not None else {"RT"},
        mode=activity_mode,
    )
    follower_counts = load_follower_counts(higgs_dir)

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
    pos = component_layout_packed(G_layout, seed=layout_seed)

    node_sizes = [120 + 60 * np.log1p(activity_counts.get(node, 0)) for node in G_draw.nodes()]
    follower_vals = np.array([follower_counts.get(node, 0) for node in G_draw.nodes()], dtype=float)
    positive_followers = follower_vals[follower_vals > 0]
    vmin = max(1.0, positive_followers.min() if len(positive_followers) else 1.0)
    vmax = max(1.0, follower_vals.max() if len(follower_vals) else 1.0)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Blues
    node_colors = cmap(norm(np.maximum(follower_vals, 1.0)))

    plt.figure(figsize=(25, 20))
    nx.draw_networkx_nodes(G_draw, pos, node_size=node_sizes, node_color=node_colors, edgecolors="black", linewidths=0.7)
    if show_labels:
        nx.draw_networkx_labels(G_draw, pos, font_size=7)

    draw_directed_edge_group(G_draw, pos, truth_2hop_only_edges, color="lightgray", style="dashed", width=0.9, alpha=0.18, arrowsize=9)
    draw_directed_edge_group(G_draw, pos, truth_1hop_edges, color="gray", style="solid", width=1.0, alpha=0.22, arrowsize=9)
    draw_directed_edge_group(G_draw, pos, hit_1hop, color="darkgreen", style="solid", width=2.5, alpha=0.95, arrowsize=16)
    draw_directed_edge_group(G_draw, pos, hit_2hop_only, color="limegreen", style="solid", width=2.4, alpha=0.92, arrowsize=16)
    draw_directed_edge_group(G_draw, pos, reverse_only, color="orange", style="solid", width=2.2, alpha=0.90, arrowsize=16)
    draw_directed_edge_group(G_draw, pos, miss, color="red", style="solid", width=1.9, alpha=0.78, arrowsize=15)

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
        "Higgs inferred-edge collisions, 4-way\n"
        f"node size = log activity, node color = followers, top {len(inferred_edges)} inferred edges"
    )
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "collision_plot": str(out_path),
        "collision_nodes_plotted": len(inferred_nodes),
        "collision_inferred_edges_plotted": len(inferred_edges),
        "collision_directed_1hop_hits": len(hit_1hop),
        "collision_directed_2hop_only_hits": len(hit_2hop_only),
        "collision_reverse_only_hits": len(reverse_only),
        "collision_misses": len(miss),
    }




def edge_set_from_df(inferred_df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(inferred_df["source"], inferred_df["target"]))


def topk_edge_set_from_df(inferred_df: pd.DataFrame, k: int) -> set[tuple[str, str]]:
    if k <= 0:
        return set()
    top = inferred_df.head(k)
    return set(zip(top["source"], top["target"]))


def jaccard_similarity(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def summarize_edge_overlaps(
    run_infos: list[dict[str, Any]],
    output_dir: Path,
    topk_values: Iterable[int],
) -> list[Path]:
    saved: list[Path] = []
    if not run_infos:
        return saved

    labels = [info["label"] for info in run_infos]
    edge_sets = {info["label"]: edge_set_from_df(info["inferred_df"]) for info in run_infos}

    counts = pd.DataFrame(index=labels, columns=labels, dtype=int)
    jacc = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for a in labels:
        for b in labels:
            inter = edge_sets[a] & edge_sets[b]
            counts.loc[a, b] = len(inter)
            jacc.loc[a, b] = jaccard_similarity(edge_sets[a], edge_sets[b])

    counts_csv = output_dir / "summary_edge_overlap_counts.csv"
    jacc_csv = output_dir / "summary_edge_overlap_jaccard.csv"
    counts.to_csv(counts_csv)
    jacc.to_csv(jacc_csv)
    saved.extend([counts_csv, jacc_csv])

    unique_rows = []
    edge_counter = Counter(e for es in edge_sets.values() for e in es)
    for label in labels:
        unique_edges = sorted(edge_sets[label] - set().union(*(edge_sets[other] for other in labels if other != label))) if len(labels) > 1 else sorted(edge_sets[label])
        unique_rows.append({
            "label": label,
            "num_edges": len(edge_sets[label]),
            "num_unique_edges": len(unique_edges),
        })
    unique_df = pd.DataFrame(unique_rows).sort_values("label")
    unique_csv = output_dir / "unique_edges_by_run.csv"
    unique_df.to_csv(unique_csv, index=False)
    saved.append(unique_csv)

    consensus_rows = []
    for edge, freq in sorted(edge_counter.items(), key=lambda x: (-x[1], x[0])):
        row = {"source": edge[0], "target": edge[1], "num_runs_present": freq}
        for label in labels:
            row[f"in_{label}"] = edge in edge_sets[label]
        consensus_rows.append(row)
    consensus_df = pd.DataFrame(consensus_rows)
    consensus_csv = output_dir / "consensus_edges_all_runs.csv"
    consensus_df.to_csv(consensus_csv, index=False)
    saved.append(consensus_csv)

    if labels:
        plt.figure(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.8)))
        plt.imshow(jacc.values.astype(float), aspect="auto")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.colorbar(label="Jaccard similarity")
        plt.title("Pairwise inferred-edge Jaccard similarity")
        plt.tight_layout()
        heatmap_path = output_dir / "summary_edge_overlap_jaccard_heatmap.png"
        plt.savefig(heatmap_path, bbox_inches="tight", dpi=200)
        plt.close()
        saved.append(heatmap_path)

    for k in topk_values:
        top_sets = {info["label"]: topk_edge_set_from_df(info["inferred_df"], k) for info in run_infos}
        counts_k = pd.DataFrame(index=labels, columns=labels, dtype=int)
        jacc_k = pd.DataFrame(index=labels, columns=labels, dtype=float)
        for a in labels:
            for b in labels:
                inter = top_sets[a] & top_sets[b]
                counts_k.loc[a, b] = len(inter)
                jacc_k.loc[a, b] = jaccard_similarity(top_sets[a], top_sets[b])

        counts_k_csv = output_dir / f"summary_top{k}_edge_overlap_counts.csv"
        jacc_k_csv = output_dir / f"summary_top{k}_edge_overlap_jaccard.csv"
        counts_k.to_csv(counts_k_csv)
        jacc_k.to_csv(jacc_k_csv)
        saved.extend([counts_k_csv, jacc_k_csv])

        unique_rows_k = []
        for label in labels:
            unique_edges_k = sorted(top_sets[label] - set().union(*(top_sets[other] for other in labels if other != label))) if len(labels) > 1 else sorted(top_sets[label])
            unique_rows_k.append({
                "label": label,
                "k": k,
                "num_topk_edges": len(top_sets[label]),
                "num_unique_topk_edges": len(unique_edges_k),
            })
        unique_k_df = pd.DataFrame(unique_rows_k).sort_values("label")
        unique_k_csv = output_dir / f"unique_top{k}_edges_by_run.csv"
        unique_k_df.to_csv(unique_k_csv, index=False)
        saved.append(unique_k_csv)

        consensus_rows_k = []
        top_counter = Counter(e for es in top_sets.values() for e in es)
        for edge, freq in sorted(top_counter.items(), key=lambda x: (-x[1], x[0])):
            row = {"source": edge[0], "target": edge[1], "num_runs_present": freq, "k": k}
            for label in labels:
                row[f"in_{label}"] = edge in top_sets[label]
            consensus_rows_k.append(row)
        consensus_k_df = pd.DataFrame(consensus_rows_k)
        consensus_k_csv = output_dir / f"consensus_top{k}_edges_all_runs.csv"
        consensus_k_df.to_csv(consensus_k_csv, index=False)
        saved.append(consensus_k_csv)

        plt.figure(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.8)))
        plt.imshow(jacc_k.values.astype(float), aspect="auto")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.colorbar(label=f"Top-{k} Jaccard similarity")
        plt.title(f"Pairwise top-{k} inferred-edge Jaccard similarity")
        plt.tight_layout()
        heatmap_k_path = output_dir / f"summary_top{k}_edge_overlap_jaccard_heatmap.png"
        plt.savefig(heatmap_k_path, bbox_inches="tight", dpi=200)
        plt.close()
        saved.append(heatmap_k_path)

    return saved

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a grid of Higgs inference experiments, evaluate each run against social truth, "
            "save per-run plus summary plots, and create collision/activity network plots."
        )
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--cascades", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--higgs-dir", type=str, default="data/higgs")
    parser.add_argument("--methods", nargs="+", default=["netrate_baseline"])
    parser.add_argument("--diffusions", nargs="+", required=True, help="Use repo-valid names like exp pl rayleigh.")
    parser.add_argument("--horizons", nargs="+", required=True, type=float)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--thr", type=float, default=1e-8)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--top-k-max", type=int, default=2000)
    parser.add_argument("--matlab-faithful", action="store_true")
    parser.add_argument("--force-relative", action="store_true")
    parser.add_argument("--top-k-inferred-plot", type=int, default=2000)
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--activity-kinds", nargs="+", default=["RT"])
    parser.add_argument("--activity-mode", type=str, default="actor", choices=["actor", "any"])
    parser.add_argument("--top-k-overlap", nargs="+", type=int, default=list(TOP_K_OVERLAP_DEFAULT))
    return parser.parse_args()


def resolve_path(base_root: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_root / p).resolve()


def main() -> None:
    args = parse_args()
    repo_root = project_root(args.repo_root)
    ensure_repo_on_path(repo_root)

    cascades_path = resolve_path(repo_root, args.cascades)
    output_dir = resolve_path(repo_root, args.output_dir)
    higgs_dir = resolve_path(repo_root, args.higgs_dir)

    relative_cascades_path = prepare_relative_cascades(cascades_path, force=args.force_relative)
    relative_df = pd.read_csv(relative_cascades_path)
    max_time = float(pd.to_numeric(relative_df["infection_time"], errors="raise").max())

    experiments = build_experiment_grid(
        cascades_path=relative_cascades_path,
        output_dir=output_dir,
        methods=args.methods,
        diffusion_types=args.diffusions,
        horizons=args.horizons,
        l1=args.l1,
        thr=args.thr,
        solver=args.solver,
        matlab_faithful=args.matlab_faithful,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_root": str(repo_root),
                "cascades_path": str(cascades_path),
                "relative_cascades_path": str(relative_cascades_path),
                "max_relative_infection_time": max_time,
                "reverse_truth": REVERSE_TRUTH,
                "experiments": [
                    {**asdict(exp), "cascades_path": str(exp.cascades_path), "out_path": str(exp.out_path)}
                    for exp in experiments
                ],
            },
            f,
            indent=2,
        )

    summary_rows: list[dict[str, Any]] = []
    run_infos: list[dict[str, Any]] = []
    print(f"Repo root: {repo_root}")
    print(f"Raw cascades: {cascades_path}")
    print(f"Relative cascades: {relative_cascades_path}")
    print(f"Max relative infection_time T = {max_time}")
    print(f"Number of runs = {len(experiments)}")

    for exp in experiments:
        print("\n=== Running", exp.label, "===")
        print("Method:", exp.method)
        print("Params:", exp.params)

        run_inference(exp.method, exp.cascades_path, exp.out_path, exp.params, repo_root)
        run_dir = exp.out_path.parent
        inferred_df, _ = load_inferred_edges(exp.out_path)

        per_run_rows: list[dict[str, Any]] = []
        for truth_type in TRUTH_TYPES:
            metrics_df, metrics_summary = evaluate_one_truth(
                inferred_path=exp.out_path,
                truth_type=truth_type,
                higgs_dir=higgs_dir,
                reverse_truth=REVERSE_TRUTH,
                top_k_max=args.top_k_max,
                repo_root=repo_root,
            )
            metrics_csv = run_dir / f"eval_{truth_type}_precision_at_k.csv"
            plot_png = run_dir / f"eval_{truth_type}_precision_at_k.png"
            metrics_df.to_csv(metrics_csv, index=False)
            save_precision_plot(metrics_df, plot_png, title=f"{exp.label} vs {truth_type}")
            row = {
                "method": exp.method,
                "label": exp.label,
                "cascades_path": str(cascades_path),
                "relative_cascades_path": str(relative_cascades_path),
                "inferred_path": str(exp.out_path),
                "metrics_csv": str(metrics_csv),
                "metrics_plot": str(plot_png),
                "type_diffusion": exp.params.get("type_diffusion"),
                "horizon": exp.params.get("horizon"),
                "l1": exp.params.get("l1"),
                "thr": exp.params.get("thr"),
                "solver": exp.params.get("solver"),
                "matlab_faithful": exp.params.get("matlab_faithful", False),
            }
            row.update(metrics_summary)
            per_run_rows.append(row)
            print(f"Saved evaluation for {exp.label} / {truth_type}:", metrics_csv, plot_png)

        collision_plot_path = run_dir / "network_collision_activity.png"
        collision_summary = save_collision_activity_plot(
            inferred_path=exp.out_path,
            out_path=collision_plot_path,
            higgs_dir=higgs_dir,
            repo_root=repo_root,
            reverse_truth=REVERSE_TRUTH,
            top_k_inferred=args.top_k_inferred_plot,
            show_labels=args.show_labels,
            activity_allowed_kinds=set(args.activity_kinds) if args.activity_kinds else None,
            activity_mode=args.activity_mode,
        )
        print("Saved collision/activity plot:", collision_plot_path)

        run_infos.append(
            {
                "label": exp.label,
                "method": exp.method,
                "run_dir": run_dir,
                "inferred_path": exp.out_path,
                "inferred_df": inferred_df,
            }
        )

        for row in per_run_rows:
            row.update(collision_summary)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "top_edges_json" in summary_df.columns:
        summary_df["top_edges_json"] = summary_df["top_edges_json"].apply(json.dumps)

    summary_csv = output_dir / "summary_runs.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\nSaved summary:", summary_csv)

    for overlap_path in summarize_edge_overlaps(run_infos, output_dir, args.top_k_overlap):
        print("Saved overlap output:", overlap_path)

    for plot_path in save_comparison_plots(summary_df, output_dir):
        print("Saved summary plot:", plot_path)


if __name__ == "__main__":
    main()
