from __future__ import annotations
from pathlib import Path
import os, json
from typing import Sequence, Union
from matplotlib.figure import Figure
import networkx as nx

COLOR_POS = "#d62728"
COLOR_NEG = "#1f77b4"
COLOR_CUT_EDGE = "#2ca02c"
COLOR_INTRA_EDGE = "#cccccc"

GraphInput = Union[nx.Graph, str, Path]

here = Path(__file__).resolve().parent
candidates = [
        Path.cwd() / "results" / "cuts",            
        here.parent.parent / "results" / "cuts",
        here.parent.parent / "results" / "tables",
]

GRAPH_DIR = next((p for p in candidates if p.is_dir()),
                          candidates[-1])

RESULTS_DIR = here.parent.parent / "results" / "tables"


def _to_nx_graph(graph: GraphInput) -> nx.Graph:
    """ Returns a graph object given different kinds of inputs """
    if isinstance(graph, nx.Graph):
        return graph
    if isinstance(graph, (str, Path)):
        import json
        with Path(graph).open() as f:
            payload = json.load(f)
        g = nx.Graph()
        g.add_nodes_from(range(int(payload["n"])))
        g.add_edges_from([(int(u), int(v)) for u, v in payload["edges"]])
        return g

def _assignment_to_sets(
    assignment: Sequence[int], n: int
) -> tuple[set, set]:
    if len(assignment) != n:
        raise ValueError(
            f"Assignment length {len(assignment)} != n={n}"
        )
    pos, neg = set(), set() # for positive and negative
    for i, x in enumerate(assignment):
        x = int(x)
        if x in (-1, 1):
            (pos if x == 1 else neg).add(i) # add the element to the set
        elif x in (0, 1):
            (pos if x == 1 else neg).add(i)
        else:
            raise ValueError(
                f"Bad assignment value at index {i}: {x}"
            )
    return pos, neg

def plot_cut(
    graph: GraphInput,
    assignment: Sequence[int],
    *,
    cut_size: int | None = None,
    title: str | None = None,
    graph_dir: Path | str | None = None,
    filename: str | None = None,
    layout_seed: int = 42,
    node_size: int = 500,
    font_size: int = 10,
    figsize: tuple[float, float] = (6.5, 5.5),
    show: bool = False,
    dpi: int = 150,
) -> Path:
    """
    Draw a graph coloured by a ±1 MaxCut assignment and save the PNG.

    Parameters
    ----------
    graph        : nx.Graph | JSON path | payload dict
    assignment   : ±1 (or 0/1) sequence, length == graph.n
    cut_size     : if None, computed as number of cut edges
    title        : plot title; defaults to "MaxCut — n=…, cut=…"
    graph_dir  : output directory; defaults to <package>/../../results/cuts
    filename     : explicit filename; default: cut_n{n:03d}_v{v}.png
    layout_seed  : deterministic spring_layout seed
    node_size, font_size, figsize, dpi : matplotlib styling
    show         : also display interactively (default False — file-only)

    Returns
    -------
    output_path : Path to the saved PNG.
    """
    G = _to_nx_graph(graph) # retrieve the graph
    n = G.number_of_nodes()
    pos, neg = _assignment_to_sets(assignment, n)

    # Compute the cut if not given, and figure out which edges are cut.
    if cut_size is None:
        cut = [(u, v) for u, v in G.edges() if (u in pos) != (v in pos)]
        cut = set(cut)
        cut_size = len(cut)

    out_dir = Path(graph_dir) if graph_dir is not None else GRAPH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(out_dir)
    if filename is None:
        filename = f"cut_n{n:03d}_v{cut_size}.png"
    out_path = out_dir / filename

    # Layout — spring_layout needs a connected graph; fall back to circular.
    if nx.is_connected(G):
        pos_layout = nx.spring_layout(G, seed=layout_seed)
    else:
        pos_layout = nx.circular_layout(G)

    # Node colours per side; nodes 0..n-1 in case graph has been renumbered.
    node_color = [
        COLOR_POS if i in pos else COLOR_NEG
        for i in range(n)
    ]

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot()

    nx.draw_networkx_nodes(
        G, pos_layout, ax=ax,
        node_color=node_color, node_size=node_size,
        edgecolors="black", linewidths=0.8,
    )
    # Cut edges: dashed green; internal edges: thin grey.
    nx.draw_networkx_edges(
        G, pos_layout, ax=ax,
        edgelist=list(cut),
        edge_color=COLOR_CUT_EDGE, width=2.2, style="dashed",
    )
    # for edges not in the cut
    nx.draw_networkx_edges(
        G, pos_layout, ax=ax,
        edgelist=[e for e in G.edges() if e not in cut],
        edge_color=COLOR_INTRA_EDGE, width=0.6,
    )
    nx.draw_networkx_labels(
        G, pos_layout, ax=ax, font_size=font_size,
        font_color="white",
    )

    ax.set_title(title or f"MaxCut — n={n}, cut={cut_size}")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    return out_path

def plot_all_cuts(
    graphs_with_assignments,
    *,
    graph_dir: Path | str | None = None,
    layout_seed: int = 42,
) -> list[Path]:
    """
    Helper for batching: pass [(graph, assignment), ...] and save one PNG each.

    Returns the list of saved file paths in the same order.
    """
    paths = []
    for graph, assignment in graphs_with_assignments:
        paths.append(
            plot_cut(
                graph,
                assignment,
                graph_dir=graph_dir,
                layout_seed=layout_seed,
            )
        )
    return paths

def save_results(results: dict, filename: str, OUT_DIR: str = ""):
    file_path = RESULTS_DIR / filename
    
    with open(file_path, "w") as file:
        json.dump(results, file)
        file.close()    
        
def retrieve_graphs():
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "data" / "graphs",            
        here.parent.parent / "data" / "graphs",
    ]
    
    graphs_dir = next((p for p in candidates if p.is_dir()),
                            candidates[-1])
        
    return sorted(p for p in graphs_dir.glob("graph_n*.json"))