import numpy as np
import cvxpy as cp
from typing import Iterable, Union
from pathlib import Path
import time
from src.common.utility import plot_cut, save_results
from src.common.graphs import load_graph


def goemans_williamson(edges: Iterable[tuple[int, int]], n :int, num_rounds: int = 200, seed: int = 0):
    """
    Goemans Williamson MaxCut.
        Inputs:
            - edges: List[int, int] of edges between nodes
            - n: int number of nodes in the graph
            
        Outputs:
            - best_cut: number of edges cut
            - best_assignment: best nodes assignment
            - SDO-optimum: optimal approx ratio is ~0.87
    """
    edges = list(edges)
    rng = np.random.default_rng(seed=seed)
    
    # build the adjacency matrix from edges
    adj_mat = [[0] * n for _ in range(n)]
    
    for (i, j) in edges:
        adj_mat[i][j] = 1
        adj_mat[j][i] = 1
    
    X = cp.Variable((n, n), symmetric=True) # create a symmetric matrix
    constraints = [X >> 0] # force the matrix to be Positive Semidefinite
    
    for i in range(n):
        constraints.append(X[i][i] == 1)
        
    # adj_mat is reduntant here. We keep it with the ambition of moving to weighted graphs.
    expression = cp.Maximize(sum(adj_mat[i][j] * (1 - X[i][j]) for i, j in edges) / 2)
    
    # define the convex optimization problem passing the constraints defined
    problem = cp.Problem(expression, constraints=constraints)
    valid_statuses = [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
    solved = False
    for solver in (cp.MOSEK, cp.SCS, cp.CLARABEL):
        try:
            problem.solve(solver=solver, verbose=False)
            
            # check if the problem status was one of the accepted ones
            if problem.status in valid_statuses:
                solved = True
                break
        except (cp.SolverError, Exception):
            continue
    if not solved:
        raise RuntimeError(
            "No SDP solver worked. Install Mosek (academic) or SCS."
        )    
        
    sdp_value = problem.value
        
    # recover eigenvectors from the Gram Matrix X
    vals, V = np.linalg.eigh(X.value)
    V = V @ np.diag(np.sqrt(np.clip(vals, 0, None)))
    
    # draw num_rounds hyperplaces and operate as following
    #       if y_i + g > 0 assign +1 to the vertex else assign -1
    best_x, best_cut = None, -1
    for _ in range(num_rounds):
        g = rng.standard_normal(n) # standard normal distribution with mu = 0, std = 1
        x = np.where(V @ g >= 0, 1.0, -1.0) # where we take each row of V and multiply it by g randomly sampled
        cut = sum(1 for (i, j) in edges if x[i] != x[j])
        if(cut > best_cut):
            best_cut = cut
            best_x = x
            
    return best_cut, best_x, sdp_value

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "data" / "graphs",            
        here.parent.parent / "data" / "graphs",
    ]
    
    graphs_dir = next((p for p in candidates if p.is_dir()),
                            candidates[-1])
        
    files = sorted(p for p in graphs_dir.glob("graph_n*.json"))
    
    results = {}
    results["n"] = []
    results["max_cut"] = []
    results["time"] = []
    results["sdp"] = []
    
    for graph in files:
            edges, n = load_graph(graph)
            
            start = time.time()
            cut, a, sdp = goemans_williamson(edges, n, 200)
            elapsed = (time.time() - start)
            
            plot_cut(graph, a, title=f"Goemans-Williamson - n={n} cut={cut}", filename=f"GW-n{n:03d}_v{cut}")
            
            print(f"Found the Maximum Cut with the Goemans-Williamson Algorithm on the graph with {n} nodes.")
            results["n"].append(n)
            results["max_cut"].append(cut)
            results["time"].append(elapsed)
            results["sdp"].append(sdp)
            
    save_results(results, "Goemans-Williamson.json")