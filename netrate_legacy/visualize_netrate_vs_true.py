# scripts/netrate/visualize_netrate_vs_true.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("TkAgg")  # comment out if you prefer the default backend
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.netrate.utils_netrate import project_root

# -------- paths (root is MAIS/) ----------
ROOT = project_root()
# Ground-truth directed edges (two columns = tail, head)
TRUE_EDGES = ROOT / "data" / "m-input" / "verona" / "raj-full-edges.csv"
# NetRate output with columns: source,target,beta  (directed)
INFERRED = ROOT / "data" / "output" / "model" / "netrate_result_true.csv"

# -------- config ----------
SEED = 42
DIRECTED_TRUE = True          # set True if the ground truth is directed
LABEL_NODES = True            # show node names
LABEL_TOP_N = 0               # 0 = label ALL nodes; else label top-N by degree

# Node spacing controls (most important)
NODE_SPACING_K = 2.0          # larger -> more distance between nodes (e.g., 1.5–3.0)
LAYOUT_ITER = 180             # more iterations -> smoother layout
LAYOUT_SCALE = 2.0            # overall layout scale

# Edge curvature (to separate overlapping arrows)
RAD_TRUE = 0.20               # curvature for true (background) arrows
RAD_PRED = 0.35               # curvature for predicted (foreground) arrows

# -------- load true graph ----------
true_df = pd.read_csv(TRUE_EDGES, dtype=str)
if {"vertex1", "vertex2"}.issubset(true_df.columns):
    u_col, v_col = "vertex1", "vertex2"
else:
    # fallback: first two columns
    u_col, v_col = true_df.columns[:2]

G_true = nx.DiGraph() if DIRECTED_TRUE else nx.Graph()
for _, r in true_df.iterrows():
    u, v = str(r[u_col]), str(r[v_col])
    if u != v:
        G_true.add_edge(u, v)

# -------- load inferred directed edges ----------
inf_df = pd.read_csv(INFERRED)
inf_df["source"] = inf_df["source"].astype(str)
inf_df["target"] = inf_df["target"].astype(str)
if "beta" not in inf_df.columns:
    inf_df["beta"] = 1.0

# -------- select top-K directed predictions for fair overlay ----------
# K equals the number of true directed edges.
K = G_true.number_of_edges()
inf_df = inf_df.sort_values("beta", ascending=False).reset_index(drop=True)
if K > 0 and len(inf_df) > K:
    inf_show = inf_df.head(K).copy()
else:
    inf_show = inf_df.copy()

# Build predicted directed graph (keep strongest duplicate)
G_inf = nx.DiGraph()
for _, r in inf_show.iterrows():
    u, v, w = r["source"], r["target"], float(r["beta"])
    if u == v:
        continue
    if G_inf.has_edge(u, v):
        G_inf[u][v]["weight"] = max(G_inf[u][v]["weight"], w)
    else:
        G_inf.add_edge(u, v, weight=w)

# -------- combined graph & spaced layout ----------
G_all = nx.DiGraph()
G_all.add_nodes_from(set(G_true.nodes()) | set(G_inf.nodes()))
G_all.add_edges_from(G_true.edges())
G_all.add_edges_from(G_inf.edges())

# Increase node spacing using spring_layout parameters
pos = nx.spring_layout(
    G_all,
    seed=SEED,
    k=NODE_SPACING_K,        # key spacing control
    iterations=LAYOUT_ITER,
    scale=LAYOUT_SCALE
)

# -------- helpers for labels ----------
def choose_labels(G, top_n=0):
    """Return a dict of node->label. top_n=0 means label all nodes."""
    if not LABEL_NODES:
        return {}
    if top_n and top_n > 0 and top_n < G.number_of_nodes():
        deg = dict(G.degree())
        top_nodes = sorted(deg, key=deg.get, reverse=True)[:top_n]
        return {n: n for n in top_nodes}
    return {n: n for n in G.nodes()}

# -------- overlay: draw ----------
plt.figure(figsize=(11.5, 11.0))

# 1) nodes
nx.draw_networkx_nodes(
    G_all, pos,
    node_size=140, node_color="skyblue", edgecolors="k", linewidths=0.5
)

# 2) true directed edges in faint gray (background)
nx.draw_networkx_edges(
    G_true, pos,
    arrows=True, arrowstyle="-|>", arrowsize=11,
    edge_color="lightgray", width=1.1, alpha=0.35,
    connectionstyle=f"arc3,rad={RAD_TRUE}"
)

# 3) predicted directed edges (colored), with clearer separation
true_dir = set(G_true.edges())
pred_dir = list(G_inf.edges())
edge_colors = ["green" if (u, v) in true_dir else "red" for (u, v) in pred_dir]

# scale widths softly by beta for visibility
weights = np.array([G_inf[u][v].get("weight", 1.0) for u, v in pred_dir], dtype=float)
if weights.size:
    wmin, wmax = weights.min(), weights.max()
    if wmax > wmin:
        widths = 1.3 + 1.7 * (weights - wmin) / (wmax - wmin)
    else:
        widths = np.full_like(weights, 1.9)
else:
    widths = np.array([])

nx.draw_networkx_edges(
    G_inf, pos,
    edgelist=pred_dir,
    arrows=True, arrowstyle="-|>", arrowsize=13,
    edge_color=edge_colors,
    width=list(widths) if widths.size else 1.9,
    alpha=0.96,
    connectionstyle=f"arc3,rad={RAD_PRED}"
)

# 4) labels (node names)
labels_overlay = choose_labels(G_all, LABEL_TOP_N)
if labels_overlay:
    nx.draw_networkx_labels(G_all, pos, labels=labels_overlay, font_size=8)

plt.title(
    "Directed NetRate vs Directed Verona\n"
    "Green = predicted & correct • Red = predicted false positive • Gray = true (missed)"
)
plt.axis("off")

out_overlay = ROOT / "data" / "output" / "model" / "verona_netrate_directed_overlay.png"
plt.tight_layout()
plt.savefig(out_overlay, dpi=240, bbox_inches="tight")
print(f"Saved overlay figure to: {out_overlay}")

# -------- true-only directed figure ----------
plt.figure(figsize=(10.5, 10.0))
nx.draw_networkx_nodes(
    G_true, pos,
    node_size=140, node_color="skyblue", edgecolors="k", linewidths=0.5
)
nx.draw_networkx_edges(
    G_true, pos,
    arrows=True, arrowstyle="-|>", arrowsize=11,
    edge_color="gray", width=1.6, alpha=0.85,
    connectionstyle=f"arc3,rad={RAD_TRUE}"
)

labels_true = choose_labels(G_true, LABEL_TOP_N)
if labels_true:
    nx.draw_networkx_labels(G_true, pos, labels=labels_true, font_size=8)

plt.title("True Verona Network (directed)")
plt.axis("off")

out_true = ROOT / "data" / "output" / "model" / "verona_true_directed.png"
plt.tight_layout()
plt.savefig(out_true, dpi=240, bbox_inches="tight")
print(f"Saved true directed network to: {out_true}")
# plt.show()  # enable if you want an interactive window
