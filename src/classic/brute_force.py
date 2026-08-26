import itertools 
import json
import time
import os
import networkx as nx
from pathlib import Path
from typing import Iterable, Union
from src.common.utility import plot_cut, plot_all_cuts, save_results, retrieve_graphs
from src.common.graphs import load_graph

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

if __name__ == "__main__":
    files = retrieve_graphs()
    
    results = {
        "n": [],
        "max_cut": [],
        "time": [],
    }
    
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
        