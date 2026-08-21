import itertools 
import json
import time
import os
import networkx as nx
from pathlib import Path
from typing import Iterable, Union
from project.src.common.utility import plot_cut, plot_all_cuts, save_results

import cvxpy as cp
import numpy as np

GraphLike = Union[nx.Graph, str, Path]

def brute_force_maxcut(edges: Iterable[tuple[int, int]], n: int) -> tuple[int, list[int]]:
    """
        Exact MaxCut by exhaustive enumeration over all 2^n assignments.
            
        edges : iterable of (int, int), edge endpoints
        n : int, number of nodes
    
        Returns
        -------
        best_cut : int
        best_a : 
    """
    assignments = itertools.product((-1, 1), repeat=n) # all possible subdivisions of a n-nodes graph
    edges = [(int(u), int(v)) for (u, v) in edges]
    best_cut = -1
    best_a = None
    
    for bits in assignments:
        a = list(bits)        
        cut = sum(1 for (i, j) in edges if a[i] != a[j])
            
        if cut > best_cut:
            best_cut = cut
            best_a = a
            
    if best_a is None:
        raise ValueError("brute_force_maxcut received no edges or empty graph")
    
    return best_cut, best_a

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
                    f"Payload dict must have 'n' and 'edges' keys; got {list(obj)}"
                )
            n = int(object["n"])
            edges = [(int(u), int(v)) for u, v in object["edges"]]
            return edges, n
        
    raise TypeError("Unsupported Graph Type")

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "data" / "graphs",            
        here.parent.parent / "data" / "graphs",
    ]
    
    graphs_dir = next((p for p in candidates if p.is_dir()),
                          candidates[-1])
        
    files = sorted(p for p in graphs_dir.glob("graph_n*.json"))
    
    results = {}
    results["n"] = []
    results["max_cut"] = []
    results["time"] = []
    
    for graph in files:
        edges, n = load_graph(graph)
        
        if n < 25:
            start = time.time()
            cut, a = brute_force_maxcut(edges, n)
            elapsed = (time.time() - start)
            # TODO: implement a graphic way of visualizing the cut
            plot_cut(graph, a)

            print(f"Solved graph with {n} nodes.")
            results["n"].append(n)
            results["max_cut"].append(cut)
            results["time"].append(elapsed)
        else:
            print(f"Graph too big (n = {n})to compute it with brute force.")
            
    save_results(results, filename="Brute-Force.json")
        