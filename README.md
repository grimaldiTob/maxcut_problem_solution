# MaxCut algorithms on 3-regular graphs

Benchmarking classical and quantum solvers for the Maximum Cut problem on 3-regular (cubic) graphs of sizes $n \in \{8, 10, 12, 14, 18, 20, 24, 30, 36\}$.

Main goal of the project is exploring the limits of quantum advantage. The expected outcome of this repo is showing how quantum technologies have a practical
and useful application in these complex settings.

## Solvers

- **Brute force** (`src/classic/brute_force.py`): exact enumeration over all $2^n$ partition states. Evaluated for $n < 25$.
- **Goemans-Williamson** (`src/classic/goemans_williamson.py`): SDP relaxation using CVXPY with random hyperplane rounding.
- **QAOA** (`src/quantum/`): quantum approximate optimization using custom and Qiskit ansatz circuits.

## Project layout

- `data/graphs/`: generated 3-regular graph instances in JSON edge-list format.
- `src/common/`: graph generators, Hamiltonian builders, and cut plotting utilities.
- `src/classic/`: classical baseline implementations.
- `src/quantum/`: QAOA ansatz construction and optimization.
- `results/`: cut visualizations (`results/cuts/`) and metric summaries (`results/tables/`).
