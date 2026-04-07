from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gzip


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r", encoding="utf-8")


def _find_social_file(higgs_dir: Path) -> Path:
    candidates = [
        higgs_dir /"higgs-activity_time.txt.gz",
        higgs_dir / "higgs-social_network.edgelist.gz",
        higgs_dir / "social_network.edgelist",
        higgs_dir / "social_network.edgelist.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find social network file in {higgs_dir}")


def load_social_truth_edges(higgs_dir: Path, reverse: bool = False) -> set[tuple[str, str]]:
    path = _find_social_file(higgs_dir)
    edges: set[tuple[str, str]] = set()

    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            u, v = str(parts[0]), str(parts[1])

            if reverse:
                u, v = v, u

            if u != v:
                edges.add((u, v))

    return edges


def build_two_hop_truth(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    out_neighbors: dict[str, set[str]] = defaultdict(set)
    for u, v in edges:
        out_neighbors[u].add(v)

    within_two_hops: set[tuple[str, str]] = set(edges)

    for u, mids in out_neighbors.items():
        for mid in mids:
            for v in out_neighbors.get(mid, set()):
                if u != v:
                    within_two_hops.add((u, v))

    return within_two_hops


def get_truth_edges(higgs_dir: Path, truth_type: str, reverse: bool = False) -> set[tuple[str, str]]:
    if truth_type == "social_1hop":
        return load_social_truth_edges(higgs_dir, reverse=reverse)

    if truth_type == "social_2hop":
        one_hop = load_social_truth_edges(higgs_dir, reverse=reverse)
        return build_two_hop_truth(one_hop)

    raise ValueError(
        f"Unsupported truth_type='{truth_type}'. "
        f"Use 'social_1hop' or 'social_2hop'."
    )