from qiskit.circuit.library import QAOAAnsatz
from qiskit import QuantumCircuit
from src.common.graphs import load_graph
from pathlib import Path
from typing import Iterable, Sequence
from src.common.hamiltonian import maxcut_coefficients
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector
import matplotlib.pyplot as plt

def initialize_params(reps: int):
    """ Helper function used to define an array of parameters that will be used in the circuit. """
    gammas = ParameterVector("gamma", reps)
    betas = ParameterVector("beta", reps)
    
    return gammas, betas


def build_ansatz_manual(edges: Iterable[tuple[int, int]],
                        n: int,
                        gammas: Sequence[float],
                        betas: Sequence[float],
                        reps: int = 1):
    """ My proper version of building the ansatz relative to the Max Cut problem """
    h, J, pauli_ops = maxcut_coefficients(edges, n)
    edge_list = [(int(u), int(v)) for u, v in edges]
    
    qc = QuantumCircuit(n)

    for qub in range(n):
        qc.h(qub) # adds an hadamart gate applied to the n-th qubit

    for layer in range(reps):        
        g = gammas[layer]
        b = betas[layer]
        
        # since all the h_i = 0 in the max cut problem this loop can be omitted
        for qub in range(n):
            qc.rz(2* g * h[qub], qub)
        
        for u, v in edge_list:
            qc.cx(u, v)
            qc.rz(2 * g * J[u][v], v)
            qc.cx(u, v)
                
        for qub in range(n):
            qc.rx(2* b, qub)
    
    return qc
    
def build_ansatz(edges: Iterable[tuple[int, int]], n: int, reps: int = 1) -> QAOAAnsatz:
    """
    Standard QAOA Ansatz using Qiskit's built-in QAOAAnsatz library class.
    Parameters
    edges : Graph edge list (0-indexed).
    n : Number of qubits (graph vertices).
    reps : Number of alternating cost and mixer layers (depth p).

    Returns QAOAAnsatz object
    """
    _, _, pauli_ops = maxcut_coefficients(edges, n)
    cost_operator = SparsePauliOp.from_sparse_list(pauli_ops, num_qubits=n)
    
    return QAOAAnsatz(cost_operator=cost_operator, reps=reps) # reps parameter to tune ???

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "data" / "graphs",            
        here.parent.parent / "data" / "graphs",
    ]
    
    graphs_dir = next((p for p in candidates if p.is_dir()),
                          candidates[-1])
        
    files = sorted(p for p in graphs_dir.glob("graph_n*.json"))
    
    for graph in files:
        edges, n = load_graph(graph)
        
        qiskit_ansatz = build_ansatz(edges=edges, n=n)
        manual_ansatz = build_ansatz_manual(edges=edges, n=n)
        
        if n < 10:
            manual_ansatz.draw("mpl")
            plt.show()
        
        print(f"Qiskit QAOAAnsatz parameters: {qiskit_ansatz.parameters}")
        print(f"Manual QAOAAnsatz parameters: {manual_ansatz.parameters}")
        print(f"Manual circuit depth: {manual_ansatz.depth()}")        
        