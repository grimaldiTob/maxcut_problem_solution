import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Iterable, Sequence
import scipy.optimize

from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler

from src.common.hamiltonian import maxcut_coefficients
from src.quantum.ansatz import build_ansatz, build_ansatz_manual, initialize_params
from src.common.utility import plot_cut, save_results, retrieve_graphs
from src.common.graphs import load_graph
from src.common.optimization import Wrapper

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
               shots: int = 256,
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
    _, _, pauli_ops = maxcut_coefficients(edges, n)
    cost_op = SparsePauliOp.from_sparse_list(pauli_ops, num_qubits=n)
    
    if manual_ansatz:
        gammas, betas = initialize_params(reps=reps)
        ansatz = build_ansatz_manual(edges, n, gammas, betas, reps)
    else:
        ansatz = build_ansatz(edges, n, reps=reps)
        
    estimator = Estimator() # we need to estimate the epxectation value of the Hamiltonian
    # generates uniform parameters in between [0, 1(
    x0 = rng.uniform(0, 1.0, size=2*reps)
    
    # first parameters sets the optimization value of the transpilation, 2 is fine for now
    pass_manager = generate_preset_pass_manager(2, AerSimulator())
    isa_circuit = pass_manager.run(ansatz)
    
    def cond_fun(vals):
        curr = vals[0]
        last4 = vals[1:]
        avg_vals = np.sum(last4) / last4.size
        return np.abs(curr - avg_vals)
    
    def objective(params: np.ndarray) -> float:
        # job = estimator.run([(ansatz, cost_op, params)])
        job = estimator.run([(isa_circuit, cost_op, params)])
        result = job.result()[0]
        return float(result.data.evs)
    
    wrapper = Wrapper(
        obj_fun=objective,
        cond_fun=cond_fun,
        threshold=1e-3
    )
    
    optimization = scipy.optimize.minimize(
        fun=wrapper.objective,
        x0=x0,
        method=optimizer_method,
        options={"maxiter": maxiter},
        callback=wrapper.callback # function called at the end of each iteration for early stopping
    )
    
    # sampler -> perform measurements = shots
    sampler = Sampler()
    qc_measured = ansatz.assign_parameters(optimization.x) # assign the parameters from optimization
    qc_measured.measure_all() # attach measurement operator to all qubits
    
    sample_job = sampler.run([qc_measured], shots=shots) # sample the circuit for `shots` times
    results = sample_job.result()[0]
    counts = results.data.meas.get_counts()
    
    
    best_cut = -1
    best_x = None
    
    for bitstrings in counts.keys():
        assignment = bistring_assignment(bitstrings)
        cut = count_cuts(edges, assignment)
        
        if cut > best_cut:
            best_cut = cut
            best_x = assignment
            
    stats = {
        "params": optimization.x.tolist(),
        "success": bool(optimization.success),
        "reps": reps
    }
    
    return best_cut, best_x, float(optimization.fun), stats
        
if __name__ == "__main__":
    files = retrieve_graphs()
    REPS = 2
    
    results = {
        "n": [],
        "reps": [],
        "max_cut": [],
        "time": [],
        "energy_opt": [],
    }
    
    for graph in files:
        edges, n = load_graph(graph)
        
        if n < 29:
            start = time.time()
            best_cut, assignment, opt, stats = solve_qaoa(
                edges=edges,
                n=n,
                optimizer_method="COBYLA", # for now we stick with this
                reps=REPS,
                maxiter=100,
                shots=1024,
                manual_ansatz=True
            )
            elapsed = time.time() - start
            
            plot_cut(
                graph=graph,
                assignment=assignment,
                title=f"QAOA MaxCut n: {n} and cut: {best_cut}",
                filename=f"QAOA_n{n}_v{best_cut}.png"
            )
            
            print(f"[QAOA p={REPS}] Solved n={n:02d} | Best Cut: {best_cut:>2} | Time: {elapsed:.2f}s | Energy: {opt:.3f}")

            results["n"].append(n)
            results["reps"].append(REPS)
            results["max_cut"].append(best_cut)
            results["time"].append(elapsed)
            results["energy_opt"].append(opt)
        
    save_results(results, filename=f"QAOA_p{REPS}.json")