from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import Iterable, Union

def maxcut_coefficients(edges: Iterable[tuple[int, int]], n: int):
    """
        Define the h, J and pauli_terms object for the MaxCut problem.
    """
    edges = list(edges)
    
    h = np.zeros(n) # diagonal of the hamiltonian
    J = np.zeros((n, n))
    pauli_terms = []
    
    for (i, j) in edges:
        J[i][j] = 1.0
        J[j][i] = 1.0
        pauli_terms.append(("ZZ", [i, j], 1.0))
    
    return h, J, pauli_terms

