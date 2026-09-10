"""
Execute the three MaxCut solvers on the graph dataset and plot the comparison.
Run from the project root:  python -m src.exec

Outputs under results/:
    tables/3-regular/Brute-Force.json
    tables/3-regular/Goemans-Williamson.json
    tables/3-regular/QAOA_p{reps}.json
    tables/gnm_d{d}/...            (same file names, one folder per family)
    plots/3-regular/approx_ratio.png
    plots/gnm_d{d}/approx_ratio.png
"""
from __future__ import annotations

import re
import time

from src.classic.brute_force import brute_force_maxcut
from src.classic.goemans_williamson import goemans_williamson
from src.common.graphs import GNM_TARGET_DEGREES, load_graph
from src.common.utility import (
    plot_approximation_ratio,
    plot_execution_times,
    retrieve_graphs,
    save_results,
)
from src.quantum.qaoa import solve_qaoa

N_EXACT_LIMIT = 24 # brute force and QAOA stop at this size
REPS_LIST = (1, 2) # QAOA depths for the 3-regular family
SHOTS = 1024 # measurement shots for the final QAOA sampling


def instance_index(graph_path) -> int:
    """Extract the instance index i from a gnm filename graph_n012_i1.json (0 for 3-regular)."""
    match = re.search(r"_i(\d+)\.json$", graph_path.name)
    return int(match.group(1)) if match else 0


def run_3_regular():
    """Benchmark the 3-regular family: one graph per size, QAOA p in (1, 2)."""
    family = "3-regular"
    files = retrieve_graphs(family=family)

    brute_results = {"n": [], "i": [], "max_cut": [], "time": []}
    gw_results = {"n": [], "i": [], "max_cut": [], "time": [], "sdp": []}
    qaoa_results = {
        reps: {"n": [], "i": [], "max_cut": [], "time": [], "energy_opt": [],
               "reps": reps, "shots": SHOTS}
        for reps in REPS_LIST
    }

    for graph in files:
        edges, n = load_graph(graph)
        i = instance_index(graph)

        # Goemans-Williamson: polynomial, runs on every size
        start = time.time()
        gw_cut, _, sdp = goemans_williamson(edges, n, num_rounds=200)
        gw_results["n"].append(n)
        gw_results["i"].append(i)
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
        brute_results["i"].append(i)
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
            qaoa_results[reps]["i"].append(i)
            qaoa_results[reps]["max_cut"].append(qaoa_cut)
            qaoa_results[reps]["energy_opt"].append(energy)
            qaoa_results[reps]["time"].append(time.time() - start)
            print(f"[QAOA p={reps}]   n={n:02d} | cut: {qaoa_cut:>2} | energy: {energy:.3f}")

    # save results for the algorithms that where run
    save_results(brute_results, filename="Brute-Force.json", family=family)
    save_results(gw_results, filename="Goemans-Williamson.json", family=family)
    for reps in REPS_LIST:
        save_results(qaoa_results[reps], filename=f"QAOA_p{reps}.json", family=family)

    plot_approximation_ratio(family=family)
    plot_execution_times(family=family, vis_gw=True)


def run_gnm(target_degree: int):
    """
    Benchmark one Erdős–Rényi G(n, m) family (gnm_d{target_degree}):
    INSTANCES_PER_SIZE graphs per size, QAOA at p=1.
    """
    family = f"gnm_d{target_degree}"
    files = retrieve_graphs(family=family)

    brute_results = {"n": [], "i": [], "max_cut": [], "time": []}
    gw_results = {"n": [], "i": [], "max_cut": [], "time": [], "sdp": []}
    qaoa_results = {
        1: {"n": [], "i": [], "max_cut": [], "time": [], "energy_opt": [],
            "reps": 1, "shots": SHOTS}
    }

    for graph in files:
        edges, n = load_graph(graph)
        i = instance_index(graph)

        # Goemans-Williamson: polynomial, runs on every size
        start = time.time()
        gw_cut, _, sdp = goemans_williamson(edges, n, num_rounds=200)
        gw_results["n"].append(n)
        gw_results["i"].append(i)
        gw_results["max_cut"].append(gw_cut)
        gw_results["sdp"].append(sdp)
        gw_results["time"].append(time.time() - start)
        print(f"[GW {family}]  n={n:02d} i={i} | cut: {gw_cut:>2} | sdp bound: {sdp:.2f}")

        if n > N_EXACT_LIMIT:
            print(f"[skip] n={n} > {N_EXACT_LIMIT}: Brute Force and QAOA skipped")
            continue

        # Brute force: exact optimum, exponential time
        start = time.time()
        bf_cut, _ = brute_force_maxcut(edges, n)
        brute_results["n"].append(n)
        brute_results["i"].append(i)
        brute_results["max_cut"].append(bf_cut)
        brute_results["time"].append(time.time() - start)
        print(f"[Brute {family}]  n={n:02d} i={i} | cut: {bf_cut:>2}")

        # QAOA: p=1 only for the ER families (agreed in the plan)
        start = time.time()
        qaoa_cut, _, energy, _ = solve_qaoa(
            edges=edges,
            n=n,
            optimizer_method="COBYLA",
            reps=1,
            maxiter=100,
            shots=SHOTS,
            manual_ansatz=True,
        )
        qaoa_results[1]["n"].append(n)
        qaoa_results[1]["i"].append(i)
        qaoa_results[1]["max_cut"].append(qaoa_cut)
        qaoa_results[1]["energy_opt"].append(energy)
        qaoa_results[1]["time"].append(time.time() - start)
        print(f"[QAOA p=1 {family}]  n={n:02d} i={i} | cut: {qaoa_cut:>2} | energy: {energy:.3f}")

    save_results(brute_results, filename="Brute-Force.json", family=family)
    save_results(gw_results, filename="Goemans-Williamson.json", family=family)
    save_results(qaoa_results[1], filename="QAOA_p1.json", family=family)

    plot_approximation_ratio(family=family)


def main():
    """Run the benchmark for every graph family in the dataset."""
    run_3_regular()
    for d in GNM_TARGET_DEGREES:
        run_gnm(target_degree=d)


if __name__ == "__main__":
    main()
