"""
Execute the three MaxCut solvers on the graph dataset and plot the comparison.
Run from the project root:  python -m src.exec

Outputs under results/:
    tables/Brute-Force.json
    tables/Goemans-Williamson.json
    tables/QAOA_p{reps}.json
    plots/approximation_ratio.png
"""
import time

from src.classic.brute_force import brute_force_maxcut
from src.classic.goemans_williamson import goemans_williamson
from src.common.graphs import load_graph
from src.common.utility import plot_approximation_ratio, retrieve_graphs, save_results, plot_execution_times
from src.quantum.qaoa import solve_qaoa

N_EXACT_LIMIT = 24 # brute force and QAOA stop at this size
REPS_LIST = (1, 2) # QAOA depths to benchmark
SHOTS = 1024 # measurement shots for the final QAOA sampling


def main():
    files = retrieve_graphs()

    brute_results = {"n": [], "max_cut": [], "time": []}
    gw_results = {"n": [], "max_cut": [], "time": [], "sdp": []}
    qaoa_results = {
        reps: {"n": [], "max_cut": [], "time": [], "energy_opt": [],
               "reps": reps, "shots": SHOTS}
        for reps in REPS_LIST
    }

    for graph in files:
        edges, n = load_graph(graph)

        # Goemans-Williamson: polynomial, runs on every size
        start = time.time()
        gw_cut, _, sdp = goemans_williamson(edges, n, num_rounds=200)
        gw_results["n"].append(n)
        gw_results["max_cut"].append(gw_cut)
        gw_results["sdp"].append(sdp)
        gw_results["time"].append(time.time() - start)
        print(f"[GW]        n={n:02d} | cut: {gw_cut:>2} | sdp bound: {sdp:.2f}")

        if n > N_EXACT_LIMIT:
            print(f"[skip] n={n} > {N_EXACT_LIMIT}: Brute Force and QAOA skipped")
            continue

        # Brute force: exact optimum, exponential time
        start = time.time()
        bf_cut, _ = brute_force_maxcut(edges, n)
        brute_results["n"].append(n)
        brute_results["max_cut"].append(bf_cut)
        brute_results["time"].append(time.time() - start)
        print(f"[Brute]     n={n:02d} | cut: {bf_cut:>2}")

        # QAOA: one optimization run per depth
        for reps in REPS_LIST:
            start = time.time()
            qaoa_cut, _, energy, _ = solve_qaoa(
                edges=edges,
                n=n,
                optimizer_method="COBYLA",
                reps=reps,
                maxiter=100,
                shots=SHOTS,
                manual_ansatz=True,
            )
            qaoa_results[reps]["n"].append(n)
            qaoa_results[reps]["max_cut"].append(qaoa_cut)
            qaoa_results[reps]["energy_opt"].append(energy)
            qaoa_results[reps]["time"].append(time.time() - start)
            print(f"[QAOA p={reps}]   n={n:02d} | cut: {qaoa_cut:>2} | energy: {energy:.3f}")

    save_results(brute_results, filename="Brute-Force.json")
    save_results(gw_results, filename="Goemans-Williamson.json")
    for reps in REPS_LIST:
        save_results(qaoa_results[reps], filename=f"QAOA_p{reps}.json")

    plot_approximation_ratio()
    plot_execution_times(vis_gw=True)


if __name__ == "__main__":
    main()
