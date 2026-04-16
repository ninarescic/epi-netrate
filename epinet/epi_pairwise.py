from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"cascade_id", "node_id", "infection_time"}


@dataclass(frozen=True)
class PairEvent:
    """One ordered within-cascade parent candidate.

    Attributes
    ----------
    cascade_id:
        Identifier of the cascade where the event pair was observed.
    source:
        Candidate parent node id (earlier in the cascade).
    target:
        Candidate child node id (later in the cascade).
    dt:
        Positive time gap t_target - t_source.
    source_recency_rank:
        0 means the most recent earlier node, 1 the second most recent, etc.
    source_popularity:
        Normalized proxy popularity in [0, 1], computed from source appearances
        across cascades. Larger values give a slightly fatter time tail.
    """

    cascade_id: str
    source: str
    target: str
    dt: float
    source_recency_rank: int
    source_popularity: float


@dataclass(frozen=True)
class PairwiseConfig:
    """Configuration for pairwise epidemiological network inference.

    Parameters
    ----------
    beta_fast:
        Fast decay rate for likely recent contagion.
    beta_slow:
        Slow decay rate for delayed visibility / resurfacing.
    mixture_rho:
        Base mass assigned to the slow component. Actual per-source slow mass is
        modulated by source popularity when popularity scaling is enabled.
    background_mu:
        Small baseline mass for unexplained activations.
    l1:
        Soft-thresholding applied after each M-step. Larger values yield a
        sparser network.
    max_lag:
        Optional maximum allowed time gap for candidate parents.
    topk_recent_parents:
        Optional cap on how many most-recent earlier nodes are considered for
        each target in a cascade.
    min_edge_support:
        Minimum expected count required for an edge to be kept in the final CSV.
    max_iter:
        Number of EM-style refinement steps.
    tol:
        Early stopping tolerance on the maximum absolute edge-weight change.
    use_popularity_tail:
        If True, sources that appear in more cascades get slightly more weight in
        the slow visibility tail.
    popularity_power:
        Nonlinearity applied to normalized popularity.
    epsilon:
        Small numerical stabilizer.
    """

    beta_fast: float = 1.0 / 3600.0
    beta_slow: float = 1.0 / 86400.0
    mixture_rho: float = 0.15
    background_mu: float = 1e-6
    l1: float = 1e-4
    max_lag: float | None = None
    topk_recent_parents: int | None = 50
    min_edge_support: float = 0.25
    max_iter: int = 25
    tol: float = 1e-6
    use_popularity_tail: bool = True
    popularity_power: float = 1.0
    epsilon: float = 1e-12

    def validate(self) -> None:
        if self.beta_fast <= 0 or self.beta_slow <= 0:
            raise ValueError("beta_fast and beta_slow must be positive.")
        if not (0.0 <= self.mixture_rho <= 1.0):
            raise ValueError("mixture_rho must lie in [0, 1].")
        if self.background_mu < 0:
            raise ValueError("background_mu must be nonnegative.")
        if self.l1 < 0:
            raise ValueError("l1 must be nonnegative.")
        if self.topk_recent_parents is not None and self.topk_recent_parents <= 0:
            raise ValueError("topk_recent_parents must be positive when provided.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass
class InferenceResult:
    """Container for inferred edges and training diagnostics."""

    edges: pd.DataFrame
    diagnostics: dict[str, float | int]


def _require_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Cascade CSV is missing required columns {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )


def load_cascades(cascades_path: str | Path) -> pd.DataFrame:
    """Load cascades and keep only the earliest infection per node per cascade.

    The Higgs cascade builder can include repeated node appearances within a
    cascade or synthetic seed rows. For pairwise inference we want a clean
    first-entry event for each node in each cascade.
    """

    cascades_path = Path(cascades_path)
    df = pd.read_csv(cascades_path)
    _require_columns(df)

    work = df.copy()
    work["cascade_id"] = work["cascade_id"].astype(str)
    work["node_id"] = work["node_id"].astype(str)
    work["infection_time"] = pd.to_numeric(work["infection_time"], errors="coerce")
    work = work.dropna(subset=["infection_time"])

    work = work.sort_values(
        ["cascade_id", "infection_time", "node_id"],
        ascending=[True, True, True],
    )

    # Keep only the first time each node appears inside a cascade.
    work = work.drop_duplicates(subset=["cascade_id", "node_id"], keep="first")
    work = work.reset_index(drop=True)
    return work


def _compute_source_popularity(df: pd.DataFrame) -> dict[str, float]:
    """Proxy popularity from how often a node appears across distinct cascades."""

    counts = df.groupby("node_id")["cascade_id"].nunique().astype(float)
    if counts.empty:
        return {}
    lo = float(counts.min())
    hi = float(counts.max())
    if hi <= lo:
        return {str(node): 0.0 for node in counts.index}
    return {str(node): float((cnt - lo) / (hi - lo)) for node, cnt in counts.items()}


def build_pair_events(
    cascades: pd.DataFrame,
    *,
    max_lag: float | None = None,
    topk_recent_parents: int | None = None,
    use_popularity_tail: bool = True,
) -> list[PairEvent]:
    """Build ordered candidate parent-child pairs from cascades.

    For every target node i in a cascade, every earlier node j is a candidate
    parent. Candidates can be restricted to a maximum lag and/or the most recent
    earlier nodes.
    """

    popularity = _compute_source_popularity(cascades) if use_popularity_tail else {}
    events: list[PairEvent] = []

    for cascade_id, group in cascades.groupby("cascade_id", sort=False):
        group = group.sort_values(["infection_time", "node_id"]).reset_index(drop=True)
        nodes = group["node_id"].tolist()
        times = group["infection_time"].astype(float).tolist()

        for tgt_pos in range(1, len(nodes)):
            tgt = nodes[tgt_pos]
            tgt_time = times[tgt_pos]
            candidates: list[tuple[str, float]] = []

            for src_pos in range(tgt_pos - 1, -1, -1):
                src = nodes[src_pos]
                dt = tgt_time - times[src_pos]
                if dt <= 0:
                    continue
                if max_lag is not None and dt > max_lag:
                    continue
                candidates.append((src, dt))
                if topk_recent_parents is not None and len(candidates) >= topk_recent_parents:
                    break

            # candidates were collected from most recent backward, so rank is natural.
            for rank, (src, dt) in enumerate(candidates):
                events.append(
                    PairEvent(
                        cascade_id=str(cascade_id),
                        source=str(src),
                        target=str(tgt),
                        dt=float(dt),
                        source_recency_rank=rank,
                        source_popularity=float(popularity.get(str(src), 0.0)),
                    )
                )

    return events


def _effective_rho(event: PairEvent, config: PairwiseConfig) -> float:
    rho = config.mixture_rho
    if not config.use_popularity_tail:
        return rho
    scaled_pop = event.source_popularity ** config.popularity_power
    # Popular sources get slightly more slow-tail mass, but keep rho bounded.
    return min(1.0, rho * (0.5 + 0.5 * scaled_pop) * 2.0)


def pair_kernel(event: PairEvent, config: PairwiseConfig) -> float:
    """Two-timescale delay kernel for candidate parent-child timing."""

    rho = _effective_rho(event, config)
    dt = event.dt
    fast = np.exp(-config.beta_fast * dt)
    slow = np.exp(-config.beta_slow * dt)
    return float((1.0 - rho) * fast + rho * slow)


def _initialize_edge_weights(events: Iterable[PairEvent], config: PairwiseConfig) -> dict[tuple[str, str], float]:
    """Initialize weights from raw support normalized by exposure."""

    counts: dict[tuple[str, str], float] = defaultdict(float)
    exposure: dict[tuple[str, str], float] = defaultdict(float)
    for event in events:
        edge = (event.source, event.target)
        kern = pair_kernel(event, config)
        counts[edge] += 1.0
        exposure[edge] += kern

    weights: dict[tuple[str, str], float] = {}
    for edge, cnt in counts.items():
        denom = exposure[edge] + config.epsilon
        weights[edge] = max(config.epsilon, cnt / denom)
    return weights


def infer_pairwise_network_from_df(
    cascades: pd.DataFrame,
    *,
    config: PairwiseConfig | None = None,
) -> InferenceResult:
    """Infer a directed network from ordered within-cascade pairs.

    The algorithm is a lightweight EM-style procedure:
    1. Build ordered candidate parent-child pair events.
    2. Score candidates with current edge weight * delay kernel.
    3. Normalize into soft parent responsibilities per target event.
    4. Update nonnegative edge weights from expected counts / exposure.
    5. Soft-threshold to encourage sparsity.
    """

    config = config or PairwiseConfig()
    config.validate()

    events = build_pair_events(
        cascades,
        max_lag=config.max_lag,
        topk_recent_parents=config.topk_recent_parents,
        use_popularity_tail=config.use_popularity_tail,
    )
    if not events:
        raise RuntimeError("No ordered pair events were built from the cascades.")

    weights = _initialize_edge_weights(events, config)

    # Group events by target observation inside each cascade, because those
    # candidates compete to explain one observed infection time.
    grouped: dict[tuple[str, str], list[PairEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.cascade_id, event.target)].append(event)

    latest_max_change = np.nan
    last_expected_total = np.nan
    last_expected_counts: dict[tuple[str, str], float] = {}
    opportunity_counts: dict[tuple[str, str], float] = defaultdict(float)
    mean_dt_sums: dict[tuple[str, str], float] = defaultdict(float)
    for event in events:
        edge = (event.source, event.target)
        opportunity_counts[edge] += 1.0
        mean_dt_sums[edge] += event.dt

    for _ in range(config.max_iter):
        expected_counts: dict[tuple[str, str], float] = defaultdict(float)
        exposure: dict[tuple[str, str], float] = defaultdict(float)
        background_mass_total = 0.0

        for candidates in grouped.values():
            numerators: list[tuple[PairEvent, tuple[str, str], float]] = []
            denom = config.background_mu
            for event in candidates:
                edge = (event.source, event.target)
                kern = pair_kernel(event, config)
                exposure[edge] += kern
                score = weights.get(edge, config.epsilon) * kern
                numerators.append((event, edge, score))
                denom += score

            if denom <= config.epsilon:
                background_mass_total += 1.0
                continue

            background_mass_total += config.background_mu / denom
            for _event, edge, score in numerators:
                resp = score / denom
                expected_counts[edge] += resp

        max_change = 0.0
        new_weights: dict[tuple[str, str], float] = {}
        expected_total = 0.0

        for edge, exp_cnt in expected_counts.items():
            expected_total += exp_cnt
            denom = exposure[edge] + config.epsilon
            raw = exp_cnt / denom
            updated = max(0.0, raw - config.l1)
            new_weights[edge] = updated
            max_change = max(max_change, abs(updated - weights.get(edge, 0.0)))

        # Keep previously seen edges that went to exactly zero out of the active set.
        weights = {edge: w for edge, w in new_weights.items() if w > 0.0}
        latest_max_change = max_change
        last_expected_total = expected_total
        last_expected_counts = dict(expected_counts)
        if max_change < config.tol:
            break

    records = []
    for edge, weight in weights.items():
        if weight <= 0.0:
            continue
        src, tgt = edge
        opportunity = opportunity_counts.get(edge, 0.0)
        support = last_expected_counts.get(edge, 0.0)
        mean_dt = mean_dt_sums.get(edge, 0.0) / opportunity if opportunity > 0 else np.nan
        records.append(
            {
                "source": src,
                "target": tgt,
                "weight": float(weight),
                "support": float(support),
                "opportunities": float(opportunity),
                "mean_dt": float(mean_dt),
            }
        )

    edges = pd.DataFrame.from_records(records)
    if edges.empty:
        edges = pd.DataFrame(columns=["source", "target", "weight", "support", "opportunities", "mean_dt"])
    else:
        edges = edges[edges["support"] >= config.min_edge_support].copy()
        edges = edges.sort_values(["weight", "support", "opportunities"], ascending=False).reset_index(drop=True)

    diagnostics = {
        "num_cascades": int(cascades["cascade_id"].nunique()),
        "num_nodes": int(cascades["node_id"].nunique()),
        "num_events": int(len(events)),
        "num_edges": int(len(edges)),
        "background_mass_total": float(background_mass_total),
        "expected_total": float(last_expected_total) if np.isfinite(last_expected_total) else np.nan,
        "final_max_change": float(latest_max_change) if np.isfinite(latest_max_change) else np.nan,
    }
    return InferenceResult(edges=edges, diagnostics=diagnostics)


def infer_pairwise_network(
    cascades_path: str | Path,
    out_path: str | Path,
    *,
    config: PairwiseConfig | None = None,
) -> InferenceResult:
    """Load cascades, infer the network, and write a CSV edge list."""

    cascades = load_cascades(cascades_path)
    result = infer_pairwise_network_from_df(cascades, config=config)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.edges.to_csv(out_path, index=False)
    return result


__all__ = [
    "PairEvent",
    "PairwiseConfig",
    "InferenceResult",
    "load_cascades",
    "build_pair_events",
    "pair_kernel",
    "infer_pairwise_network_from_df",
    "infer_pairwise_network",
]
