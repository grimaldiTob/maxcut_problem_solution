from __future__ import annotations
from pathlib import Path
import os, json
from typing import Sequence, Union
import matplotlib.pyplot as plt
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
PLOTS_DIR = here.parent.parent / "results" / "plots"


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

def plot_approximation_ratio(results_dir=RESULTS_DIR, outfile="approx_ratio.png", family: str = ""):
    """
    Plot cut / optimum per solver for one graph family.

    Reads the result tables of ``results/tables/<family>/`` and saves the
    figure to ``results/plots/<family>/``. The 0.6924 QAOA p=1 bound only
    holds for 3-regular graphs, so the line is drawn on that family only;
    every family gets the m/2 random-cut baseline (ratio m/(2·opt)).
    """
    tables_dir = results_dir / family if family else results_dir
    plots_dir = PLOTS_DIR / family if family else PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    # load the dictionaries from the results file
    with open(tables_dir / "Brute-Force.json") as f:
        bf = json.load(f)
    with open(tables_dir / "Goemans-Williamson.json") as f:
        gw = json.load(f)

    optimum = dict(zip(bf["n"], bf["max_cut"])) # take the exact optimum

    # random-cut baseline: a coin-flip assignment cuts m/2 edges on
    # average, so its ratio is (m/2) / optimum. m comes from the graph
    # files themselves (averaged over instances for gnm families).
    m_sum: dict[int, float] = {}
    m_count: dict[int, int] = {}
    for p in retrieve_graphs(family or "3-regular"):
        payload = json.load(open(p))
        n = int(payload["n"])
        m_sum[n] = m_sum.get(n, 0.0) + len(payload["edges"]) / 2
        m_count[n] = m_count.get(n, 0) + 1
    m_per_n = {n: m_sum[n] / m_count[n] for n in m_sum}

    fig = Figure(figsize=(7, 5));
    ax = fig.add_subplot()

    ax.axhline(1.0, color="k", lw=0.8, label="Brute force (optimum)")

    # the bounds are checked and proved for both the algorithms
    ax.axhline(0.878, color="gray", ls="-.", lw=1, label="GW guarantee (0.878)")
    if family == "3-regular":
        ax.axhline(0.6924, color="gray", ls="--", lw=1, label="QAOA p=1 bound, 3-regular")

    def ratio(series, label, style):
        ns   = [n for n in series["n"] if n in optimum]
        vals = [c / optimum[n] for n, c in zip(series["n"], series["max_cut"]) if n in optimum]
        ax.plot(ns, vals, style, label=label)

    baseline_ns = sorted(n for n in m_per_n if n in optimum)
    if baseline_ns:
        ax.plot(baseline_ns, [m_per_n[n] / optimum[n] for n in baseline_ns],
                ":", color="dimgray", lw=1.2, label="random cut (m/2)")

    ratio(gw, "Goemans–Williamson", "s-")

    # consider the maxcut problems solved so far
    for tag in ("p1", "p2"):
        qaoa_file = tables_dir / f"QAOA_{tag}.json"
        if qaoa_file.is_file():  # ER families only have p=1
            ratio(json.load(open(qaoa_file)), f"QAOA {tag}", "o--")

    ax.set_xlabel("# nodes n");
    ax.set_ylabel("cut / optimum")
    ax.set_ylim(0.6, 1.05); ax.legend(loc="lower left")
    fig.savefig(plots_dir / outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
def plot_execution_times(results_dir = RESULTS_DIR, filename: str = "exec_time.png", vis_gw : bool = False, family: str = ""):
    """
    Plot the per-solver execution times of one graph family.

    Reads the result tables of ``results/tables/<family>/`` and saves the
    figure to ``results/plots/<family>/``. If vis_gw is set the function
    also produces a separate plot showing the Goemans-Williamson time per n.
    """
    tables_dir = results_dir / family if family else results_dir
    plots_dir = PLOTS_DIR / family if family else PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    # open the .json files and retrieve the dict
    with open(tables_dir / "Brute-Force.json") as f:
        bf = json.load(f)
        
    with open(tables_dir / "Goemans-Williamson.json") as f:
        gw = json.load(f)
        
    with open(tables_dir / "QAOA_p1.json") as f:
        qaoa_p1 = json.load(f)       

        
    with open(tables_dir / "QAOA_p2.json") as f:
        qaoa_p2 = json.load(f)         
    
    fig = Figure(figsize=(7, 5));
    ax = fig.add_subplot()  
    
    ax.plot(bf["n"], bf["time"], c="black", marker="o", linestyle="--", label="Brute Force", linewidth=0.7)
    ax.plot(gw["n"], gw["time"], c="red", marker="o", linestyle="--", label="Goemans-Williamson", linewidth=0.7)
    ax.plot(qaoa_p1["n"], qaoa_p1["time"], c="green", marker="o", linestyle="--", label="QAOA p=1", linewidth=0.7)
    ax.plot(qaoa_p2["n"], qaoa_p2["time"], c="blue", marker="o", linestyle="--", label="QAOA p=2", linewidth=0.7)
    ax.set_xlabel("# nodes")
    ax.set_ylabel("Execution time (s)")
    
    ax.legend()
    ax.grid(True)
    fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    if vis_gw:
        fig = Figure(figsize=(7, 5));
        ax = fig.add_subplot()  
        gw_ms = [x*1000 for x in gw["time"]]
        
        ax.plot(gw["n"], gw_ms, c="red", marker="o", linestyle="--", label="Goemans-Williamson", linewidth=0.7)
        ax.set_xlabel("# nodes")
        ax.set_ylabel("Execution time G-W (ms)")
        ax.legend()
        ax.grid(True)
        fig.savefig(plots_dir / "gw_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
def save_results(results: dict, filename: str, family: str = ""):
    """
    Save a results dict as JSON.

    With `family` set, results land in ``results/tables/<family>/`` so that
    different graph families never overwrite each other's tables; the empty
    default keeps the historical flat layout.
    """
    out_dir = RESULTS_DIR / family if family else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename

    with open(file_path, "w") as file:
        json.dump(results, file)
        file.close()

def retrieve_graphs(family: str = "3-regular"):
    """
    Obtain a list with the filenames of the graphs .json files of one
    graph family, stored in data/graphs/<family>/.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "data" / "graphs",
        here.parent.parent / "data" / "graphs",
    ]

    graphs_dir = next((p for p in candidates if p.is_dir()),
                            candidates[-1])

    family_dir = graphs_dir / family
    if not family_dir.is_dir():
        raise FileNotFoundError(
            f"{family_dir} (expected e.g. '3-regular' or 'gnm_d3')"
        )

    return sorted(p for p in family_dir.glob("graph_n*.json"))