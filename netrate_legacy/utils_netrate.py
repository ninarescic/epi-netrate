"""
utils_netrate.py
----------------
Common helper functions for NetRate experiment scripts.
Used by:
  - run_experiment_netrate.py
  - peek_cascades.py
  - peek_results.py
  - visualize_netrate_vs_true.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# --- PATH HELPERS ------------------------------------------------------------

def project_root():
    """Return the root directory of the MAIS project."""
    return Path(__file__).resolve().parents[2]


def data_dir(subpath=None):
    """Return path to MAIS/data/output/model (optionally joined with subpath)."""
    root = project_root() / "data" / "output" / "model"
    return root if subpath is None else root / subpath


# --- DATA HELPERS ------------------------------------------------------------

def load_cascades(csv_path):
    """Load NetRate cascades (CSV) as DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} infections across {df['cascade_id'].nunique()} cascades.")
    return df


def load_results(csv_path):
    """Load NetRate results (edges and betas)."""
    df = pd.read_csv(csv_path)
    df.sort_values("beta", ascending=False, inplace=True)
    print(f"Loaded {len(df)} inferred edges.")
    return df


# --- GRAPH HELPERS -----------------------------------------------------------

def load_true_graph(edge_path):
    """Load the true Verona contact graph."""
    df = pd.read_csv(edge_path)
    G = nx.from_pandas_edgelist(df, "vertex1", "vertex2")
    print(f"True graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def compare_graphs(G_true, G_inferred, top_k=None):
    """
    Compare true vs inferred edges.
    Returns precision, recall, and overlap sets.
    """
    true_edges = {tuple(sorted(e)) for e in G_true.edges()}
    inferred_edges = {tuple(sorted(e)) for e in G_inferred.edges()}

    if top_k:
        inferred_edges = set(list(inferred_edges)[:top_k])

    overlap = inferred_edges & true_edges
    precision = len(overlap) / len(inferred_edges) if inferred_edges else 0
    recall = len(overlap) / len(true_edges) if true_edges else 0

    print(f"TP overlap={len(overlap)}  Precision={precision:.3f}  Recall={recall:.3f}")
    return precision, recall, overlap


def plot_beta_distribution(df, bins=40):
    """Simple β histogram."""
    plt.hist(df["beta"], bins=bins)
    plt.title("Distribution of inferred transmission rates (β)")
    plt.xlabel("β")
    plt.ylabel("Count")
    plt.show()
