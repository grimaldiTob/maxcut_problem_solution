"""
Graph generators for MaxCut/QAOA studies.

Two families, each persisted to its own subfolder of ``data/graphs/``:

    "3-regular"  cubic graphs, one per size (nx.random_regular_graph)
    "gnm_d{d}"   Erdős–Rényi G(n, m) graphs with a random edge count drawn around target degree d
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Union

import networkx as nx

SIZES = (8, 10, 12, 14, 18, 20, 24, 30, 36)
# Layout seed — used for visualisations only, not for graph generation.
LAYOUT_SEED = 42

CUBIC_FAMILY = "3-regular"
GNM_TARGET_DEGREES = (2, 3, 4, 5)  # one gnm_d{d} family per target degree
INSTANCES_PER_SIZE = 3 # generate this number of graphs for each gnm graph type

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "graphs"
MANIFEST_NAME = "manifest.json"
SCHEMA = "maxcut-qaoa/edge-list-v1"

GraphLike = Union[nx.Graph, str, Path]

def make_cubic(n: int, seed: int) -> nx.Graph:
    """
    Create a 3-regular (cubic) graph with `n` nodes.

    Parameters
    n : int
        Number of nodes. Must be even and > 6 for a 3-regular graph
        to exist.
    seed : int
        RNG seed for reproducibility.
    """
    if n <= 6 or n % 2 != 0:
        raise ValueError(
            f"3-regular simple graph requires even n > 6, got n={n}"
        )
    return nx.random_regular_graph(3, n, seed=seed)


def sample_num_edges(n: int, target_degree: int, rng: random.Random) -> int:
    """
    Draw the edge count m of one G(n, m) instance:
    m ~ Uniform(0.75, 1.25) * d * n / 2, i.e. ±25% around the edge count
    of a d-regular graph.
    """
    expected = target_degree * n / 2
    return rng.randint(int(0.75 * expected), int(1.25 * expected))


def make_gnm(n: int, target_degree: int, seed: int, connected: bool = True) -> tuple[nx.Graph, int, int]:
    """
    Create an Erdős–Rényi G(n, m) graph with a random edge count.

    Parameters
    n : int
        Number of nodes.
    target_degree : int
        Average degree the edge count is drawn around (±25% window).
    seed : int
        RNG seed: the whole redraw sequence is reproducible from it.
    connected : bool
        If True, redraw until the graph is connected.

    Returns
    (graph, m, attempts)
        The graph, its realised edge count, and how many draws it took.
    """
    rng = random.Random(seed)
    attempts = 0
    
    # repeat until the graph is connected (just if the connected flag is True)
    while True:
        attempts += 1
        m = sample_num_edges(n, target_degree, rng) # it already creates the graph with the 25% intervals
        graph = nx.gnm_random_graph(n, m, seed=rng.getrandbits(32))
        if not connected or nx.is_connected(graph):
            return graph, m, attempts


def _sorted_edges(graph: nx.Graph) -> list[list[int]]:
    edges = [
        sorted([int(u), int(v)])
        for u, v in graph.edges()
    ]
    edges.sort()
    return edges

def graph_to_payload(graph: nx.Graph, n: int, seed: int) -> dict:
    """Serialise a 3-regular graph to our edge-list JSON schema (k = 3)."""
    return {
        "schema": SCHEMA,
        "n": int(n),
        "k": 3,
        "seed": int(seed),
        "edges": _sorted_edges(graph),
    }


def gnm_to_payload(graph: nx.Graph, n: int, seed: int,
                   target_degree: int, attempts: int) -> dict:
    """Serialise a G(n, m) graph, recording realised-density metadata."""
    m = graph.number_of_edges()
    return {
        "schema": SCHEMA,
        "n": int(n),
        "target_degree": int(target_degree),
        "average_degree": round(2 * m / n, 3),
        "seed": int(seed),
        "connected": bool(nx.is_connected(graph)),
        "attempts": int(attempts),
        "edges": _sorted_edges(graph),
    }


def payload_to_graph(payload: dict) -> nx.Graph:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unknown schema: {payload.get('schema')!r}")
    g = nx.Graph()
    g.add_nodes_from(range(payload["n"]))
    g.add_edges_from(payload["edges"])
    return g

def generate_all(sizes: Iterable[int] = SIZES, out_dir: Path = DATA_DIR, graph_family: str = CUBIC_FAMILY) -> dict[int, Path]:
    """
    Generate one 3-regular graph per size in ``sizes`` and persist each
    to ``out_dir / graph_family`` as JSON. Returns a mapping ``{n: filepath}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    
    out_dir = out_dir / graph_family
    for n in sizes:
        # Per-size seed = 100 + n, matching the original script.
        seed = 100 + n
        graph = make_cubic(n, seed=seed)
        payload = graph_to_payload(graph, n=n, seed=seed)
        path = out_dir / f"graph_n{n:03d}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written[n] = path
    return written


def generate_all_gnm(sizes: Iterable[int] = SIZES, out_dir: Path = DATA_DIR,
                     target_degree: int = 3,
                     instances: int = INSTANCES_PER_SIZE) -> dict[tuple[int, int], Path]:
    """
    Generate `instances` random G(n, m) graphs per size for one target
    degree and persist them to ``out_dir / gnm_d{target_degree}``.
    Returns a mapping ``{(n, i): filepath}``.

    Per-instance seed = 100_000 * d + 100 * n + i, unique across all
    families, sizes and instances.
    """
    family_dir = out_dir / f"gnm_d{target_degree}"
    family_dir.mkdir(parents=True, exist_ok=True)
    written: dict[tuple[int, int], Path] = {}

    for n in sizes:
        for i in range(instances):
            seed = 100_000 * target_degree + 100 * n + i
            graph, m, attempts = make_gnm(n, target_degree, seed=seed)
            payload = gnm_to_payload(graph, n=n, seed=seed,
                                     target_degree=target_degree, attempts=attempts)
            path = family_dir / f"graph_n{n:03d}_i{i}.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            written[(n, i)] = path
    return written


def write_manifest(out_dir: Path = DATA_DIR, graph_family: str = CUBIC_FAMILY) -> Path:
    """
    Write the manifest of one family into its own subfolder, listing the
    graph files actually present there (so it never goes stale).
    """
    family_dir = out_dir / graph_family
    family_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in family_dir.glob("graph_n*.json"))
    # size from "graph_n012.json" or "graph_n012_i0.json" → 12
    sizes = sorted({
        int(name.removeprefix("graph_n").removesuffix(".json").split("_")[0])
        for name in files
    })
    manifest = {
        "schema": SCHEMA,
        "graph_family": graph_family, # crucial to determine the kind of graph
        "count": len(files),
        "sizes": sizes,
        "files": files,
    }
    path = family_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def generate_dataset(out_dir: Path = DATA_DIR) -> tuple[dict[int, Path], Path]:
    """Convenience: write the 3-regular graphs + manifest, return (graphs_map, manifest_path)."""
    written = generate_all(out_dir=out_dir)
    manifest = write_manifest(out_dir=out_dir, graph_family=CUBIC_FAMILY)
    return written, manifest


def generate_gnm_dataset(target_degree: int, sizes: Iterable[int] = SIZES,
                         out_dir: Path = DATA_DIR,
                         instances: int = INSTANCES_PER_SIZE) -> tuple[dict[tuple[int, int], Path], Path]:
    """Write one gnm family's graphs + manifest. In this case we generate 'instances' graphs"""
    written = generate_all_gnm(sizes=sizes, out_dir=out_dir,
                               target_degree=target_degree, instances=instances)
    manifest = write_manifest(out_dir=out_dir, graph_family=f"gnm_d{target_degree}")
    return written, manifest

def load_graph(obj: GraphLike):
    """Normalise any supported input to ``(edges, n).
        At the moment it just supports the following types:
            - nx.Graph
            - str
            - Path
    """
    if(isinstance(obj, nx.Graph)):
        n = obj.number_of_nodes()
        edges = [(int(u), int(v)) for u, v in obj.edges()] 
        return edges, n
    
    if(isinstance(obj, (str, Path))):
        with Path(obj).open() as f:
            object = json.load(f)
            
            if "n" not in object or "edges" not in object:
                raise ValueError(
                    f"Payload dict must have 'n' and 'edges' keys; got {list(object)}"
                )
            n = int(object["n"])
            edges = [(int(u), int(v)) for u, v in object["edges"]]
            return edges, n
        
    raise TypeError("Unsupported Graph Type")

if __name__ == "__main__":
    project_root = DATA_DIR.parent.parent

    written, manifest = generate_dataset()
    print(f"family: {CUBIC_FAMILY}")
    for n, path in sorted(written.items()):
        print(f"  n={n:>3}  →  {path.relative_to(project_root)}")

    for d in GNM_TARGET_DEGREES:
        written, manifest = generate_gnm_dataset(target_degree=d)
        print(f"family: gnm_d{d}  (manifest: {manifest.relative_to(project_root)})")
        for n in SIZES:
            rows = [json.loads(p.read_text()) for (nn, _), p in written.items() if nn == n]
            ms = sorted(len(r["edges"]) for r in rows)
            att = sorted(r["attempts"] for r in rows)
            ok = all(r["connected"] for r in rows)
            print(f"  n={n:>3}  m∈[{ms[0]}, {ms[-1]}]  attempts∈[{att[0]}, {att[-1]}]  connected={ok}")
