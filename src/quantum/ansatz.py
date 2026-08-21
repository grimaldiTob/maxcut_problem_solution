from qiskit.circuit.library import QAOAAnsatz
from qiskit import QuantumCircuit
from src.common.graphs import load_graph
from pathlib import Path
from typing import Iterable
from src.common.hamiltonian import maxcut_coefficients
from qiskit.quantum_info import SparsePauliOp

def build_ansatz_myself(edges: Iterable[tuple[int, int]], n: int):
    h, J, pauli_ops = maxcut_coefficients(edges, n)
    
    return
    
def build_ansatz(edges: Iterable[tuple[int, int]], n: int):
    h, J, pauli_ops = maxcut_coefficients(edges, n)
    cost_operator = SparsePauliOp.from_sparse_list(pauli_ops, num_qubits=n)
    
    ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=1) # reps parameter to tune ???
    ansatz.draw("mpl")

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
        
        build_ansatz(edges, n)
        
        