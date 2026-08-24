import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Iterable, Sequence

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from src.common.hamiltonian import maxcut_coefficients
from src.quantum.ansatz import build_ansatz, build_ansatz_manual, initialize_params
from src.common.utility import plot_cut, save_results, retrieve_graphs
from src.common.graphs import load_graph

def bistring_assignment(bitstring: str) -> Sequence[int]:
    # index i of the bitstring corresponds to qubit i
    return [1 if x == "0" else -1 for x in bitstring[::-1]] # iterate the bitstring the other way around

def count_cuts(edges: Iterable[tuple[int, int]], assignment: Sequence[int]):
    return sum(1 for u, v in edges if assignment[u] != assignment[v])

def solve_qaoa(edges: Iterable[tuple[int, int]],
               n:int,
               optimizer_method: str,
               reps: int = 1,
               maxiter: int = 100,
               shots: int = 1024,
               manual_ansatz: bool = False):
    """
    Run QAOA for MaxCut on a given graph.
    Parameters

    edges : list of (u, v) tuples
    n : number of nodes
    reps : depth of the QAOA circuit (p)
    optimizer_method : classical optimizer
    maxiter : maximum classical optimization iterations
    shots : number of measurement shots for final state sampling
    Returns

    best_cut : int, size of best cut found
    best_assignment : list[int], node assignment in {+1, -1}
    energy_opt : float, minimum expectation value reached
    stats : dict, runtime and convergence diagnostics
    """
    
    edges = list(edges)
    rng = np.random.default_rng(42)
    
    if manual_ansatz:
        gammas, betas = initialize_params(reps=reps)
        ansatz = build_ansatz_manual(edges, n, gammas, betas, reps)
    else:
        ansatz = build_ansatz(edges, n, reps=reps)
        
    estimator = Stateve
        
if __name__ == "__main__":
    files = retrieve_graphs()
    
    for graph in files:
        edges, n = load_graph(graph)

        qc = build_ansatz_manual(edges=edges, n=n)
        h, J, pauli_coeff = maxcut_coefficients(edges=edges, n=n)
        
        