"""Interstellar Route Planner - command-line front end.

Examples
  python3 main.py --n 300 --seed 4 --algo all --plot demo.png
  python3 main.py --n 1000 --seed 12 --algo astar --start 17 --goal 803 --hazard exponential
"""
import argparse

import numpy as np

from search import ALGOS, ALGO_LABEL, run_algo
from universe import HAZARD_MODES, generate_universe, pick_start_goal
from visualize import save_comparison


def describe_path(U, r):
    lines = []
    for a, b in zip(r.path, r.path[1:]):
        c = U.edge_cost(a, b)
        d = U.euclid(a, b)
        lines.append(f"    body {a:>5} -> {b:>5}   dist {d:7.1f}   hazard x{c / d:4.2f}   cost {c:8.1f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="number of celestial bodies (50-5000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--hazard", choices=HAZARD_MODES, default="uniform")
    ap.add_argument("--obstacles", type=float, default=0.10, help="fraction of bodies blocked")
    ap.add_argument("--algo", choices=list(ALGOS) + ["all"], default="all")
    ap.add_argument("--start", type=int, default=None, help="start body index (default: random)")
    ap.add_argument("--goal", type=int, default=None, help="goal body index (default: random)")
    ap.add_argument("--distance", choices=["any", "near", "mid", "far"], default="far",
                    help="how far apart a random start and goal are")
    ap.add_argument("--plot", default=None, help="save a map image to this path")
    ap.add_argument("--quiet", action="store_true", help="hide the hop-by-hop route")
    a = ap.parse_args()

    U = generate_universe(a.n, a.seed, a.k_min, a.k_max, a.hazard, a.obstacles)
    s, g = pick_start_goal(U, a.seed, a.distance)
    s = a.start if a.start is not None else s
    g = a.goal if a.goal is not None else g
    for name, v in (("start", s), ("goal", g)):
        if not 0 <= v < U.n or U.blocked[v]:
            raise SystemExit(f"{name} body {v} is out of range or is an obstacle")

    print(f"Universe seed={a.seed}: {U.n} bodies, {U.edge_count} routes, "
          f"{int(U.blocked.sum())} obstacles, hazard={a.hazard}")
    print(f"Start body {s}  ->  goal body {g}   (straight-line distance {U.euclid(s, g):.1f})\n")

    algos = ALGOS if a.algo == "all" else (a.algo,)
    results = {name: run_algo(name, U, s, g, record_explored=bool(a.plot)) for name in algos}
    best = min((r.cost for r in results.values() if r.found), default=None)

    print(f"{'algorithm':<10}{'found':<7}{'cost':>12}{'vs best':>9}{'hops':>6}{'expanded':>10}"
          f"{'generated':>11}{'time ms':>9}")
    for name, r in results.items():
        if r.found:
            print(f"{ALGO_LABEL[name]:<10}{'yes':<7}{r.cost:>12,.1f}{r.cost / best:>8.2f}x{r.hops:>6}"
                  f"{r.expanded:>10}{r.generated:>11}{r.time_ms:>9.2f}")
        else:
            print(f"{ALGO_LABEL[name]:<10}{'NO':<7}{'-':>12}{'-':>9}{'-':>6}{r.expanded:>10}"
                  f"{r.generated:>11}{r.time_ms:>9.2f}")
    if not a.quiet:
        for name, r in results.items():
            if r.found:
                print(f"\n{ALGO_LABEL[name]} route ({r.hops} hops, total cost {r.cost:,.1f}):")
                print(describe_path(U, r))
            else:
                print(f"\n{ALGO_LABEL[name]}: no route exists between these bodies.")
    if a.plot:
        save_comparison(U, results, s, g, a.plot,
                        f"Interstellar Route Planner - seed {a.seed}, {U.n} bodies, "
                        f"{int(U.blocked.sum())} obstacles")
        print(f"\nMap saved to {a.plot}")


if __name__ == "__main__":
    main()
