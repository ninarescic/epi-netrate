#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NetRate-style inference in Python using CVXPY.

Modes:
- default: current legacy/surrogate behavior
- --matlab-faithful: formulation closer to original Matlab/CVX NetRate

Input CSV columns:
    cascade_id,node_id,infection_time

Output CSV columns:
    source,target,beta
"""

import argparse
from collections import defaultdict
from math import inf
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


def load_cascades(path_csv: Path):
    df = pd.read_csv(path_csv)
    df["cascade_id"] = df["cascade_id"].astype(str)
    df["node_id"] = df["node_id"].astype(str)
    df["infection_time"] = df["infection_time"].astype(float)

    cascades = defaultdict(dict)
    for cid, nid, t in df[["cascade_id", "node_id", "infection_time"]].itertuples(index=False, name=None):
        cascades[cid][nid] = float(t)

    nodes = sorted(set(df["node_id"].unique().tolist()))
    return cascades, nodes


def _kernel_value(delta: float, diffusion: str):
    if diffusion == "exp":
        return delta
    if diffusion == "pl":
        return np.log(delta) if delta > 1 else None
    if diffusion == "rayleigh":
        return 0.5 * (delta ** 2)
    raise ValueError(f"Unknown diffusion model: {diffusion}")


def _hazard_weight(delta: float, diffusion: str):
    if diffusion == "exp":
        return 1.0
    if diffusion == "pl":
        val = 1.0 / delta
        return val if val < 1 else None
    if diffusion == "rayleigh":
        return delta
    raise ValueError(f"Unknown diffusion model: {diffusion}")


def build_matlab_stats(cascades, nodes, horizon: float, diffusion: str):
    node_index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    A_potential = np.zeros((N, N), dtype=float)
    A_bad = np.zeros((N, N), dtype=float)
    num_cascades = np.zeros(N, dtype=int)

    infected_events_by_target = {i: [] for i in range(N)}

    for cid, times in cascades.items():
        items = sorted(times.items(), key=lambda kv: kv[1])
        idx_ord = [node_index[n] for n, _ in items]
        val = [float(t) for _, t in items]
        infected_set = set(idx_ord)

        # infected target contributions and predecessor info
        for pos in range(1, len(idx_ord)):
            i = idx_ord[pos]
            t_i = val[pos]
            num_cascades[i] += 1
            preds = []
            for prev in range(pos):
                j = idx_ord[prev]
                delta = t_i - val[prev]
                kval = _kernel_value(delta, diffusion)
                if kval is None:
                    continue
                A_potential[j, i] += kval
                preds.append((j, delta))
            infected_events_by_target[i].append(preds)

        # non-infected target contributions up to global horizon
        for j in idx_ord:
            t_j = times[nodes[j]]
            for i in range(N):
                if i in infected_set:
                    continue
                delta = horizon - t_j
                kval = _kernel_value(delta, diffusion)
                if kval is None:
                    continue
                A_bad[j, i] += kval

    return A_potential, A_bad, num_cascades, infected_events_by_target


def infer_targets_matlab_faithful(cascades, nodes, horizon: float, diffusion: str = "exp", thr: float = 1e-8, solver: str = "SCS", seed: int = 0, verbose: bool = True):
    if seed is not None:
        np.random.seed(seed)

    N = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    A_potential, A_bad, num_cascades, infected_events = build_matlab_stats(cascades, nodes, horizon, diffusion)

    edges = []
    B_full = np.zeros((N, N), dtype=float)

    solver_map = {
        "SCS": cp.SCS,
        "ECOS": cp.ECOS,
        "CLARABEL": cp.CLARABEL,
    }
    chosen_solver = solver_map.get(solver.upper(), cp.SCS)

    for idx_i, i_name in enumerate(nodes):
        if verbose:
            print(f"[NetRate faithful] Target {idx_i+1}/{N}: {i_name}")

        if num_cascades[idx_i] == 0:
            continue

        active_parents = np.where(A_potential[:, idx_i] > 0)[0]
        if len(active_parents) == 0:
            continue

        a_hat = cp.Variable(len(active_parents), nonneg=True)
        local_pos = {gidx: k for k, gidx in enumerate(active_parents)}

        obj_terms = []
        linear_weights = A_potential[active_parents, idx_i] + A_bad[active_parents, idx_i]
        obj_terms.append(-linear_weights @ a_hat)

        for preds in infected_events[idx_i]:
            usable = []
            weights = []
            for j_global, delta in preds:
                if j_global not in local_pos:
                    continue
                h = _hazard_weight(delta, diffusion)
                if h is None:
                    continue
                usable.append(local_pos[j_global])
                weights.append(h)
            if not usable:
                continue
            hazard_expr = cp.sum(cp.multiply(np.asarray(weights, dtype=float), a_hat[usable]))
            obj_terms.append(cp.log(hazard_expr + 1e-12))

        objective = cp.Maximize(cp.sum(obj_terms))
        prob = cp.Problem(objective)

        try:
            prob.solve(solver=chosen_solver, verbose=False)
        except Exception as e:
            if verbose:
                print(f" -> Solver failed for target {i_name}: {e}")
            continue

        if a_hat.value is None:
            continue

        betas = np.asarray(a_hat.value).ravel()
        for gidx, beta in zip(active_parents, betas):
            if beta is None or beta <= thr:
                continue
            src = nodes[gidx]
            dst = i_name
            beta_f = float(beta)
            edges.append((src, dst, beta_f))
            B_full[gidx, idx_i] = beta_f

    return edges, B_full


def infer_targets_surrogate(cascades, nodes, T_end, l1=1e-2, thr=1e-8, solver="SCS", seed=0, verbose=True):
    # Existing implementation can remain here unchanged.
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser(description="NetRate in Python with optional Matlab-faithful mode.")
    ap.add_argument("--cascades", required=True, help="Path to netrate_cascades.csv")
    ap.add_argument("--out", default="netrate_result.csv", help="Output CSV for inferred edges")
    ap.add_argument("--thr", type=float, default=1e-8, help="Drop edges with beta <= thr")
    ap.add_argument("--topk", type=int, default=0, help="Keep only global top-K edges by beta")
    ap.add_argument("--no_self_loops", action="store_true")
    ap.add_argument("--save_B", default="")
    ap.add_argument("--save_nodes", default="")
    ap.add_argument("--solver", default="SCS", choices=["SCS", "ECOS", "CLARABEL"])
    ap.add_argument("--seed", type=int, default=0)

    # legacy/surrogate compatibility
    ap.add_argument("--l1", type=float, default=1e-2)

    # Matlab-faithful mode
    ap.add_argument("--matlab_faithful", action="store_true")
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--type_diffusion", default="exp", choices=["exp", "pl", "rayleigh"])

    args = ap.parse_args()

    cascades, nodes = load_cascades(Path(args.cascades))

    if args.matlab_faithful:
        if args.horizon is None:
            raise ValueError("--horizon is required in --matlab_faithful mode")
        if args.l1 not in (0, 0.0, 1e-2):
            print("[warn] --l1 is ignored in --matlab_faithful mode")
        edges, B = infer_targets_matlab_faithful(
            cascades,
            nodes,
            horizon=args.horizon,
            diffusion=args.type_diffusion,
            thr=args.thr,
            solver=args.solver,
            seed=args.seed,
            verbose=True,
        )
    else:
        T_end = {cid: max(times.values()) for cid, times in cascades.items()}
        edges, B = infer_targets_surrogate(
            cascades,
            nodes,
            T_end,
            l1=args.l1,
            thr=args.thr,
            solver=args.solver,
            seed=args.seed,
            verbose=True,
        )

    if not edges:
        print("No edges inferred.")
        return

    df = pd.DataFrame(edges, columns=["source", "target", "beta"])
    if args.no_self_loops:
        df = df[df["source"] != df["target"]]
    if args.topk and len(df) > args.topk:
        df = df.sort_values("beta", ascending=False).head(args.topk)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    if args.save_B:
        save_path = Path(args.save_B)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".npy":
            np.save(save_path, B)
        else:
            pd.DataFrame(B).to_csv(save_path, header=False, index=False)

    if args.save_nodes:
        nodes_path = Path(args.save_nodes)
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(len(nodes)), "node_id": nodes}).to_csv(nodes_path, index=False)


if __name__ == "__main__":
    main()
