from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epinet.epi_pairwise import PairwiseConfig, infer_pairwise_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer a directed Higgs network from ordered within-cascade pairs using "
            "a simple epidemiological time-delay model."
        )
    )
    parser.add_argument(
        "--cascades",
        type=Path,
        default=ROOT / "outputs" / "higgs_rt_cascades_21_50_top50.csv",
        help="Input cascade CSV with columns cascade_id,node_id,infection_time.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "higgs_rt_inferred_epi_pairwise.csv",
        help="Output inferred edge CSV.",
    )
    parser.add_argument(
        "--relative-times",
        action="store_true",
        help=(
            "Normalize infection times to start at zero inside each cascade before "
            "inference. Pairwise gaps are unchanged, so this is mainly a debugging "
            "and consistency convenience."
        ),
    )
    parser.add_argument("--beta-fast", type=float, default=1.0 / 3600.0)
    parser.add_argument("--beta-slow", type=float, default=1.0 / 86400.0)
    parser.add_argument("--mixture-rho", type=float, default=0.15)
    parser.add_argument("--background-mu", type=float, default=1e-6)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--max-lag", type=float, default=None)
    parser.add_argument("--topk-recent-parents", type=int, default=50)
    parser.add_argument("--min-edge-support", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--no-popularity-tail",
        action="store_true",
        help="Disable the slow-tail boost for popular sources.",
    )
    parser.add_argument("--popularity-power", type=float, default=1.0)
    return parser.parse_args()


def maybe_make_relative_times(cascades_path: Path) -> Path:
    df = pd.read_csv(cascades_path)
    required = {"cascade_id", "infection_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot normalize relative times because the input is missing {sorted(missing)}"
        )
    df = df.copy()
    df["infection_time"] = pd.to_numeric(df["infection_time"], errors="coerce")
    df = df.dropna(subset=["infection_time"])
    df["infection_time"] = (
        df["infection_time"]
        - df.groupby("cascade_id")["infection_time"].transform("min")
    )
    relative_path = cascades_path.with_name(cascades_path.stem + "_relative.csv")
    df.to_csv(relative_path, index=False)
    return relative_path


def main() -> None:
    args = parse_args()
    cascades_path = args.cascades

    if args.relative_times:
        cascades_path = maybe_make_relative_times(cascades_path)
        print("Saved relative-time cascade file to:", cascades_path)

    config = PairwiseConfig(
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
        mixture_rho=args.mixture_rho,
        background_mu=args.background_mu,
        l1=args.l1,
        max_lag=args.max_lag,
        topk_recent_parents=args.topk_recent_parents,
        min_edge_support=args.min_edge_support,
        max_iter=args.max_iter,
        tol=args.tol,
        use_popularity_tail=not args.no_popularity_tail,
        popularity_power=args.popularity_power,
    )

    result = infer_pairwise_network(
        cascades_path=cascades_path,
        out_path=args.out,
        config=config,
    )

    print("Saved inferred edges to:", args.out)
    print("Diagnostics:")
    for key, value in result.diagnostics.items():
        print(f"  {key}: {value}")
    if not result.edges.empty:
        print("Top inferred edges:")
        print(result.edges.head(10).to_string(index=False))
    else:
        print("No edges survived the support / sparsity thresholds.")


if __name__ == "__main__":
    main()
