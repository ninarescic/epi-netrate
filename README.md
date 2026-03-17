# epi-netrate

`epi-netrate` is a small Python project for inferring directed diffusion networks from cascade data using a NetRate-style convex optimization baseline.

The repository currently includes:
- a reusable Python wrapper in `epinet`
- a legacy inference implementation in `netrate_legacy`
- toy scripts for quick smoke tests
- preprocessing and evaluation scripts for the Higgs dataset

## What it does

Given cascade observations of the form:

- `cascade_id`
- `node_id`
- `infection_time`

the model estimates a directed weighted graph where each edge weight represents an inferred transmission rate from one node to another.

The main output is an edge list such as:

- `source`
- `target`
- `beta`

where larger `beta` means stronger inferred influence / transmission rate.

## Status

This project is currently an early-stage research / experimentation codebase.

It is runnable and packaged, but still lightweight:
- the main inference baseline works
- toy examples run end-to-end
- some scripts are experimental and dataset-specific
- the implementation is NetRate-style and uses a convex surrogate formulation

## Installation

Clone the repository and install it in editable mode:

```bash
python -m pip install -e .
```

If needed, upgrade packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

## Quick start

### 1. Generate toy cascades

```bash
python -m scripts.make_toy_cascades
```

### 2. Run baseline inference

```bash
python -m scripts.run_toy_baseline
```

### 3. Visualize / compare inferred edges

```bash
python -m scripts.visualize_toy_true_vs_netrate
```

## Python API

The main public entrypoint is:

```python
from epinet import infer_netrate_baseline

infer_netrate_baseline(
    cascades_path="data/processed/toy_cascades.csv",
    out_path="outputs/toy_inferred.csv",
    params={
        "l1": 1e-3,
        "thr": 1e-6,
        "solver": "SCS",
        "save_B": True,
        "save_nodes": True,
    },
)
```

## Input format

The baseline expects a CSV file with at least these columns:

- `cascade_id`
- `node_id`
- `infection_time`

Example:

```csv
cascade_id,node_id,infection_time
0,1,0.0
0,2,1.0
0,3,2.0
1,2,0.0
1,1,1.5
```

## Output format

The main inferred edge list is written as a CSV with columns:

- `source`
- `target`
- `beta`

Depending on options, the code can also save:
- the full inferred transmission matrix `B`
- the node-index mapping used internally

## Main parameters

Common parameters passed through `params=` include:

- `l1`: L1 regularization strength for sparsity
- `thr`: threshold for keeping inferred edges
- `topk`: optionally keep only the top-k edges
- `no_self_loops`: remove self-loops
- `save_B`: save the full inferred rate matrix
- `save_nodes`: save the node mapping
- `solver`: currently `SCS` or `ECOS`
- `seed`: random seed

Example:

```python
params = {
    "l1": 1e-3,
    "thr": 1e-6,
    "solver": "SCS",
    "no_self_loops": True,
}
```

## Repository layout

```text
epinet/             Public Python wrapper / utilities
netrate_legacy/     Legacy inference implementation
scripts/            Toy runs, evaluation, visualization
datasets/           Dataset-specific preprocessing helpers
data/               Generated local data files
outputs/            Generated inference outputs
```

## Notes on the model

This repository implements a NetRate-style baseline for diffusion network inference.

It should be treated as a practical convex optimization implementation for experiments, rather than a polished library. Some scripts are designed for local research workflows and specific datasets.

## Troubleshooting

### `ModuleNotFoundError: No module named 'epinet'`

Run scripts as modules from the repository root:

```bash
python -m scripts.make_toy_cascades
```

not:

```bash
python scripts/make_toy_cascades.py
```

### `ModuleNotFoundError: No module named 'cvxpy'`

Install dependencies:

```bash
python -m pip install -e .
```

### Solver issues

Supported solvers are currently:

- `SCS`
- `ECOS`

If one solver behaves poorly, try the other.

