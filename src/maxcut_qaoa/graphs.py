"""
3-regular graph generators for MaxCut/QAOA studies.

Persists each graph to ``data/graphs/graph_n***.json`` as an
edge-list with metadata, so the dataset is human-readable, diffable in
git, and framework-agnostic (NetworkX, rustworkx, Qiskit, etc.).

Each file looks like:
    {
      "n": 8,
      "k": 3,
      "seed": 108,
      "edges": [[0, 1], [0, 3], ...],
      "schema": "maxcut-qaoa/edge-list-v1"
    }

A combined ``data/graphs/manifest.json`` is also written so consumers
can enumerate the dataset without scanning the directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import networkx as nx

# ─── Configuration ─────────────────────────────────────────────────────
SIZES = (8, 10, 12, 14, 18, 20, 24, 30, 36)
# Layout seed — used for visualisations only, not for graph generation.
LAYOUT_SEED = 42

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "graphs"
MANIFEST_NAME = "manifest.json"
SCHEMA = "maxcut-qaoa/edge-list-v1"

def make_cubic(n: int, seed: int) -> nx.Graph:
    """
    Create a 3-regular (cubic) graph with ``n`` nodes.

    Parameters
    ----------
    n : int
        Number of nodes. Must be even and > 6 for a 3-regular graph
        to exist (Handshake lemma + simple graph constraint).
    seed : int
        RNG seed for reproducibility.
    """
    if n <= 6 or n % 2 != 0:
        raise ValueError(
            f"3-regular simple graph requires even n > 6, got n={n}"
        )
    return nx.random_regular_graph(3, n, seed=seed)


def graph_to_payload(graph: nx.Graph, n: int, seed: int) -> dict:
    """Serialise a NetworkX graph to our edge-list JSON schema."""
    edges = [
        sorted([int(u), int(v)])
        for u, v in graph.edges()
    ]
    edges.sort()  # stable, diff-friendly ordering
    return {
        "schema": SCHEMA,
        "n": int(n),
        "k": 3,
        "seed": int(seed),
        "edges": edges,
    }


def payload_to_graph(payload: dict) -> nx.Graph:
    """Inverse of func:`graph_to_payload`."""
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unknown schema: {payload.get('schema')!r}")
    g = nx.Graph()
    g.add_nodes_from(range(payload["n"]))
    g.add_edges_from(payload["edges"])
    return g


def graph_filename(n: int) -> str:
    return f"graph_n{n:03d}.json"


def generate_all(sizes: Iterable[int] = SIZES, out_dir: Path = DATA_DIR) -> dict[int, Path]:
    """
    Generate one 3-regular graph per size in ``sizes`` and persist each
    to ``out_dir`` as JSON. Returns a mapping ``{n: filepath}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    for n in sizes:
        # Per-size seed = 100 + n, matching the original script.
        seed = 100 + n
        graph = make_cubic(n, seed=seed)
        payload = graph_to_payload(graph, n=n, seed=seed)
        path = out_dir / graph_filename(n)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written[n] = path
    return written


def write_manifest(sizes: Iterable[int] = SIZES, out_dir: Path = DATA_DIR) -> Path:
    """Write a combined manifest listing all graphs in the dataset."""
    sizes = list(sizes)
    manifest = {
        "schema": SCHEMA,
        "graph_family": "3-regular",
        "count": len(sizes),
        "sizes": sizes,
        "files": [graph_filename(n) for n in sizes],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def generate_dataset(out_dir: Path = DATA_DIR) -> tuple[dict[int, Path], Path]:
    """Convenience: write graphs + manifest, return (graphs_map, manifest_path)."""
    written = generate_all(out_dir=out_dir)
    manifest = write_manifest(out_dir=out_dir)
    return written, manifest


# ─── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    written, manifest = generate_dataset()
    for n, path in sorted(written.items()):
        print(f"  n={n:>3}  →  {path.relative_to(DATA_DIR.parent.parent)}")
    print(f"Manifest: {manifest.relative_to(DATA_DIR.parent.parent)}")
