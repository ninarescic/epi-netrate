#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NetRate (exponential kernel) in Python using cvxpy.
Input:  CSV with columns: cascade_id,node_id,infection_time
Output: CSV with columns: source,target,beta   (DIRECTED: source -> target with rate beta)

Extras in this version:
- Optional save of the full directed rate matrix B (rows=source, cols=target).
- Optional save of node list (index -> node_id) for stable mapping.
- Top-K and threshold filtering on exported edges.
- Choice of solver and deterministic layout seed for reproducibility.
"""

import argparse
from collections import defaultdict
from math import inf
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp


def load_cascades(path_csv: Path):
    """
    Returns:
      cascades: dict[cid] -> dict[node] = infection_time (float)
      nodes: sorted list of all nodes seen (as strings)
      T_end: dict[cid] -> observation window (max observed infection time in that cascade)
    """
    df = pd.read_csv(path_csv)
    # normalize types
    df["cascade_id"] = df["cascade_id"].astype(str)
    df["node_id"] = df["node_id"].astype(str)
    df["infection_time"] = df["infection_time"].astype(float)

    cascades = defaultdict(dict)
    for cid, nid, t in df[["cascade_id", "node_id", "infection_time"]].itertuples(index=False, name=None):
        cascades[cid][nid] = float(t)

    nodes = sorted(set(df["node_id"].unique().tolist()))
    # observation window: last observed infection time in each cascade
    T_end = {cid: max(times.values()) for cid, times in cascades.items()}
    return cascades, nodes, T_end


def infer_targets(cascades, nodes, T_end, l1=1e-2, thr=1e-8, solver="SCS", seed=0, verbose=True):
    """
    Solve NetRate per target node (exponential transmission).
    Returns:
      edges: list of (src, dst, beta) for beta > thr
      B: full directed rate matrix (shape [N,N]) with B[src, dst] = beta_{src->dst}
    """
    if seed is not None:
        # cvxpy solvers won't be deterministic, but layout/masks will be stable
        np.random.seed(seed)

    N = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}

    # Precompute: per cascade, sorted nodes and times
    casc_nodes = {}
    for cid, times in cascades.items():
        items = sorted(times.items(), key=lambda kv: kv[1])
        casc_nodes[cid] = (np.array([n for n, _ in items], dtype=object),
                           np.array([t for _, t in items], dtype=float))

    # Collect results per-target and fill a full matrix
    edges = []
    B_full = np.zeros((N, N), dtype=float)

    # Choose solver
    solver_map = {"SCS": cp.SCS, "ECOS": cp.ECOS, "OSQP": cp.OSQP}
    chosen_solver = solver_map.get(solver.upper(), cp.SCS)

    for idx_i, i in enumerate(nodes):
        if verbose:
            print(f"[NetRate] Target {idx_i+1}/{N}: {i}")

        # Candidate parents j: any node infected before i's infection (or before T_c if i wasn't infected)
        parents = set()
        for cid, (n_arr, t_arr) in casc_nodes.items():
            t_i = cascades[cid].get(i, inf)
            limit = t_i if np.isfinite(t_i) else T_end[cid]
            # add nodes infected before 'limit'
            for n_j, t_j in zip(n_arr, t_arr):
                if t_j < limit and n_j != i:
                    parents.add(n_j)

        parents = sorted(parents)
        if not parents:
            continue  # no incoming candidates for this target

        K = len(parents)
        b = cp.Variable(K, nonneg=True)  # beta_{j->i}, j in parents

        obj_terms = []

        # Build objective (sum over cascades)
        for cid, (n_arr, t_arr) in casc_nodes.items():
            t_i = cascades[cid].get(i, inf)

            # infection times aligned to 'parents'
            t_j_vec = np.full(K, np.nan)
            for k, pj in enumerate(parents):
                t_j_vec[k] = cascades[cid].get(pj, np.nan)

            if np.isfinite(t_i):
                # i infected in this cascade
                mask = ~np.isnan(t_j_vec) & (t_j_vec < t_i)
                if not np.any(mask):
                    continue
                delta = (t_i - t_j_vec[mask])
                # Exponential NetRate (convex surrogate here):
                # Survival piece approximated by linear term sum_j beta_j * delta_j
                # Plus -log(sum_j beta_j) for the hazard (stabilized)
                lin = cp.sum(cp.multiply(b[mask], delta))
                log_term = -cp.log(cp.sum(b[mask]) + 1e-12)
                obj_terms.append(lin + log_term)
            else:
                # i NOT infected: only survival until observation window T_c
                T = T_end[cid]
                mask = ~np.isnan(t_j_vec) & (t_j_vec < T)
                if not np.any(mask):
                    continue
                delta = (T - t_j_vec[mask])
                lin = cp.sum(cp.multiply(b[mask], delta))
                obj_terms.append(lin)

        if not obj_terms:
            # Nothing informative for this target
            continue

        objective = cp.Minimize(cp.sum(obj_terms) + l1 * cp.norm1(b))
        prob = cp.Problem(objective)
        try:
            prob.solve(solver=chosen_solver, verbose=False)
        except Exception as e:
            if verbose:
                print(f"  -> Solver failed for target {i}: {e}")
            continue

        if b.value is None:
            continue

        betas = np.asarray(b.value).ravel()
        dst_idx = node_index[i]
        for pj, beta in zip(parents, betas):
            if beta is None or beta <= thr:
                continue
            src_idx = node_index[pj]
            beta_f = float(beta)
            edges.append((pj, i, beta_f))              # strings for CSV
            B_full[src_idx, dst_idx] = beta_f          # numeric matrix

    return edges, B_full


def main():
    ap = argparse.ArgumentParser(description="NetRate (exponential) in Python with cvxpy (DIRECTED output).")
    ap.add_argument("--cascades", required=True, help="Path to netrate_cascades.csv")
    ap.add_argument("--l1", type=float, default=1e-2, help="L1 sparsity weight (lambda)")
    ap.add_argument("--thr", type=float, default=1e-8, help="Drop edges with beta <= thr")
    ap.add_argument("--topk", type=int, default=0, help="Keep only global top-K edges by beta (0 = keep all)")
    ap.add_argument("--no_self_loops", action="store_true", help="Exclude i->i edges from output (recommended)")
    ap.add_argument("--out", default="netrate_result.csv", help="Output CSV for inferred edges (src,dst,beta)")
    ap.add_argument("--save_B", default="", help="Optional path to save full B matrix (.npy or .csv)")
    ap.add_argument("--save_nodes", default="", help="Optional path to save node list (index->node_id) as CSV")
    ap.add_argument("--solver", default="SCS", choices=["SCS", "ECOS", "OSQP"], help="Convex solver")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (layout/masks)")
    args = ap.parse_args()

    cascades, nodes, T_end = load_cascades(Path(args.cascades))
    print(f"Loaded {len(cascades)} cascades, {len(nodes)} nodes.")

    edges, B = infer_targets(
        cascades, nodes, T_end,
        l1=args.l1, thr=args.thr, solver=args.solver, seed=args.seed, verbose=True
    )

    if not edges:
        print("No edges inferred (all betas ~ 0). Try reducing --l1 or providing more cascades.")
        return

    # Build DataFrame and apply filters (self-loops, threshold already applied in solver stage)
    df = pd.DataFrame(edges, columns=["source", "target", "beta"])
    if args.no_self_loops:
        df = df[df["source"] != df["target"]]

    # Optional global Top-K by beta
    if args.topk and args.topk > 0 and len(df) > args.topk:
        df = df.sort_values("beta", ascending=False).head(args.topk)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} directed edges to {out_path.resolve()}")

    # Save B if requested
    if args.save_B:
        save_path = Path(args.save_B)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".npy":
            np.save(save_path, B)
        elif save_path.suffix.lower() in (".csv", ".tsv"):
            sep = "," if save_path.suffix.lower() == ".csv" else "\t"
            pd.DataFrame(B).to_csv(save_path, header=False, index=False, sep=sep)
        else:
            print(f"[warn] Unsupported B extension '{save_path.suffix}'. Use .npy/.csv/.tsv. Skipped.")
        print(f"Saved full B matrix to {save_path.resolve()}")

    # Save node list if requested
    if args.save_nodes:
        nodes_path = Path(args.save_nodes)
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(len(nodes)), "node_id": nodes}).to_csv(nodes_path, index=False)
        print(f"Saved node mapping (index->node_id) to {nodes_path.resolve()}")


if __name__ == "__main__":
    main()
