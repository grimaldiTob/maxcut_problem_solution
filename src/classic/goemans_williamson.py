import numpy as np
import cvxpy as cp
from typing import Iterable, Union

def goemans_williamson(edges: Iterable[tuple[int, int]], n :int, num_rounds: int = 200, seed: int = 0):
    """
    Goemans Williamson MaxCut.
        Inputs:
            - edges: List[int, int] of edges between nodes
            - n: int number of nodes in the graph
            
        Outputs:
            - SDO-optimum: optimal approx ratio is ~0.87
            - best_cut: number of edges cut
            - best_assignment: best nodes assignment
    """
    edges = list(edges)
    rng = np.random.default_rng(seed=seed)
    
    X = cp.Variable((n, n), symmetric=True) # create a symmetric matrix
    constraints = [X >> 0] # force the matrix to be Positive Semidefinite
    
    for i in range(n):
        constraints.append(X[i, i] == 1)
        
    expression = 0.5 * cp.sum([1 - X[i, j] for (i, j) in edges])
    
    # define the convex optimization problem passing the constraints defined
    problem = cp.Problem(cp.Maximize(expression), constraints=constraints)
    valid_statuses = [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
    sdp_opt = False
    for solver in (cp.MOSEK, cp.SCS, cp.CLARABEL):
        try:
            sdp_opt = problem.solve(solver=solver, verbose=False)
            if problem.status in valid_statuses:
                sdp_opt = True
                break
        except (cp.SolverError, Exception):
            continue
    if not sdp_opt:
        raise RuntimeError(
            "No SDP solver worked. Install Mosek (academic) or SCS."
        )    

    return