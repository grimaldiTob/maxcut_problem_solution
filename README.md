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

### Results Obtained, so far

Main objectives of the project were proving the following points:

- **Brute Force** algorithms manages to reach the optimal solution up to a point where the number of nodes gets to big `n = 24`, beyond which it becomes completely impractical.
- **Goemans-Williamson** is a great stochastic alternative to find solutions close to the optimum with high guarantees. The main goal of the project here is testing these guarantees and verify them.
- **QAOA (p = 1)** maps a quantum state to a circuit and exploits clever ansatz and parameters optimization to reach solution with almost perfect guarantees.

Actually from the result we got, it looks like as long as n stays contained the `GW` algorithm hields very precise results with much lower execution time (x750 faster than the Quantum optimization).

full numbers in `results/tables/`, cut visualizations in `results/cuts/`.

| n   | Optimal (brute force) | GW cut | QAOA p=1 cut | BF time  | GW time | QAOA time |
| --- | --------------------- | ------ | ------------ | -------- | ------- | --------- |
| 8   | 10                    | 10     | 10           | 0.0001 s | 0.019 s | 0.52 s    |
| 12  | 15                    | 15     | 15           | 0.003 s  | 0.015 s | 0.26 s    |
| 18  | 23                    | 23     | 23           | 0.25 s   | 0.028 s | 0.69 s    |
| 24  | 32                    | 32     | 32           | 20.3 s   | 0.034 s | 25.8 s    |
| 30  | —                     | 41     | —            | —        | 0.044 s | —         |
| 36  | —                     | 49     | —            | —        | 0.064 s | —         |
