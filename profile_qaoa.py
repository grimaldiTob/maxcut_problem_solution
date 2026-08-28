"""Instrumented profiling of solve_qaoa to find the actual bottleneck.

Times the three phases separately:
  1. circuit/hamiltonian construction
  2. classical optimization loop (estimator calls)
  3. final sampling
and counts estimator objective evaluations.
"""
import sys
import time

import numpy as np
import scipy.optimize

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from pathlib import Path

from src.common.hamiltonian import maxcut_coefficients
from src.common.graphs import load_graph
from src.quantum.ansatz import build_ansatz, build_ansatz_manual, initialize_params
from src.common.optimization import Wrapper

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def profile_qaoa(edges, n, manual_ansatz, reps=1, maxiter=100):
    edges = list(edges)
    _, _, pauli_ops = maxcut_coefficients(edges, n)
    cost_op = SparsePauliOp.from_sparse_list(pauli_ops, num_qubits=n)

    t0 = time.perf_counter()
    if manual_ansatz:
        gammas, betas = initialize_params(reps=reps)
        ansatz = build_ansatz_manual(edges, n, gammas, betas, reps)
    else:
        ansatz = build_ansatz(edges, n, reps=reps)
    t_build = time.perf_counter() - t0

    estimator = StatevectorEstimator()
    rng = np.random.default_rng(42)
    x0 = rng.uniform(0, 1.0, size=2 * reps)

    n_evals = [0]
    t_obj = [0.0]

    def objective(params):
        t = time.perf_counter()
        job = estimator.run([(ansatz, cost_op, params)])
        result = job.result()[0]
        n_evals[0] += 1
        t_obj[0] += time.perf_counter() - t
        return float(result.data.evs)
    
    def cond_fun(vals):
        curr = vals[0]
        last4 = vals[1:]
        avg_vals = np.sum(last4) / last4.size
        return np.abs(curr - avg_vals)
    
    wrapper = Wrapper(
        obj_fun=objective,
        cond_fun=cond_fun,
        threshold=1e-4
    )

    t0 = time.perf_counter()
    opt = scipy.optimize.minimize(
            fun=wrapper.objective,
            x0=x0,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
            callback=wrapper.callback # function called at the end of each iteration for early stopping
    )
    t_opt = time.perf_counter() - t0

    return {
        "manual": manual_ansatz,
        "n": n,
        "reps": reps,
        "n_evals": n_evals[0],
        "t_build_s": round(t_build, 3),
        "t_obj_total_s": round(t_obj[0], 3),
        "t_obj_per_eval_ms": round(1000 * t_obj[0] / max(n_evals[0], 1), 3),
        "t_opt_total_s": round(t_opt, 3),
        "energy": round(float(opt.fun), 3),
        "circuit_depth": ansatz.depth(),
        "circuit_2q_gates": ansatz.count_ops().get("cx", 0),
    }


if __name__ == "__main__":
    graphs = [
        "/home/tgrimaldi/dev/python/qiskit/project/data/graphs/graph_n008.json",
        "/home/tgrimaldi/dev/python/qiskit/project/data/graphs/graph_n014.json",
        "/home/tgrimaldi/dev/python/qiskit/project/data/graphs/graph_n020.json",
    ]

    print(f"{'ansatz':>8} {'n':>3} {'evals':>6} {'build':>7} {'obj/s':>8} {'ms/eval':>8} {'opt/s':>8} {'depth':>6} {'cx':>5}")
    for g in graphs:
        edges, n = load_graph(g)
        manual = True
        r = profile_qaoa(edges, n, manual_ansatz=manual, reps=1, maxiter=100)
        print(
            f"{'manual' if r['manual'] else 'qiskit':>8} "
            f"{r['n']:>3} {r['n_evals']:>6} {r['t_build_s']:>7} "
            f"{r['t_obj_total_s']:>8} {r['t_obj_per_eval_ms']:>8} "
            f"{r['t_opt_total_s']:>8} {r['circuit_depth']:>6} {r['circuit_2q_gates']:>5}"
        )
