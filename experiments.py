"""Batch experiment runner: every instance is run through every algorithm on the identical graph.

One factor is varied at a time around a default condition; 30 seeds per condition.
Output: results/raw_results.csv  (one row per seed x condition x algorithm)
"""
import csv
import os
import sys
import time

from search import run_algo
from universe import generate_universe, pick_start_goal

SEEDS = list(range(30))
BASE = dict(n=1000, k_min=2, k_max=6, hazard="uniform", obstacle_frac=0.10, distance="any")
CORE = ["ucs", "greedy", "astar", "bfs"]
HEUR = ["ucs", "astar_h0", "astar", "astar_w2"]
TIMING_REPEATS = 3          # runtime = best of 3 to damp Python timing noise


def conditions():
    out = []
    for n in (50, 100, 250, 500, 1000, 2500, 5000):
        out.append(("size", str(n), dict(BASE, n=n), CORE))
    for k in (2, 3, 4, 6):
        out.append(("density", f"k={k}", dict(BASE, k_min=k, k_max=k), CORE))
    for hz in ("constant", "uniform", "exponential", "heavy"):
        out.append(("hazard", hz, dict(BASE, hazard=hz), CORE))
    for ob in (0.0, 0.10, 0.20, 0.30, 0.40):
        out.append(("obstacles", f"{int(ob * 100)}%", dict(BASE, obstacle_frac=ob), CORE))
    for d in ("near", "mid", "far"):
        out.append(("distance", d, dict(BASE, distance=d), CORE))
    out.append(("heuristic", "n=1000", dict(BASE), HEUR))
    out.append(("heuristic", "n=5000", dict(BASE, n=5000), HEUR))
    return out


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "raw_results.csv")
    fields = ["experiment", "condition", "seed", "n", "algo", "found", "cost", "hops", "expanded",
              "generated", "max_frontier", "time_ms", "optimal_cost", "straight_dist", "solvable"]
    t_start = time.time()
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for exp, cond, p, algos in conditions():
            for seed in SEEDS:
                U = generate_universe(p["n"], seed, p["k_min"], p["k_max"], p["hazard"],
                                      p["obstacle_frac"])
                s, g = pick_start_goal(U, seed, p["distance"])
                res = {}
                for a in algos:
                    r = run_algo(a, U, s, g)
                    t = r.time_ms
                    for _ in range(TIMING_REPEATS - 1):
                        t = min(t, run_algo(a, U, s, g).time_ms)
                    r.time_ms = t
                    res[a] = r
                opt = res["ucs"].cost if res["ucs"].found else float("nan")
                for a, r in res.items():
                    w.writerow(dict(experiment=exp, condition=cond, seed=seed, n=p["n"], algo=a,
                                    found=int(r.found), cost=r.cost if r.found else "",
                                    hops=r.hops, expanded=r.expanded, generated=r.generated,
                                    max_frontier=r.max_frontier, time_ms=round(r.time_ms, 4),
                                    optimal_cost=opt, straight_dist=round(U.euclid(s, g), 3),
                                    solvable=int(res["ucs"].found)))
                    rows += 1
            print(f"  done {exp:<10} {cond:<10}  ({time.time() - t_start:5.1f}s)", flush=True)
    print(f"wrote {rows} rows to {path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
