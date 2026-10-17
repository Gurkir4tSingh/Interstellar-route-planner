"""Find a representative 'greedy trap' instance and render the 4-algorithm comparison figure."""
import json, sys
import numpy as np
from search import run_algo
from universe import generate_universe, pick_start_goal
from visualize import save_comparison

N, HZ, DIST, SCAN = 300, "uniform", "far", 500
rows = []
for seed in range(SCAN):
    U = generate_universe(N, seed, hazard=HZ)
    s, g = pick_start_goal(U, seed, DIST)
    u, a, gr = (run_algo(x, U, s, g) for x in ("ucs", "astar", "greedy"))
    if not u.found:
        continue
    rows.append((gr.cost / u.cost, seed, s, g, u.expanded, a.expanded, gr.expanded, u.cost, gr.cost))
rows.sort(reverse=True)
print("scanned", SCAN, "seeds; solvable:", len(rows))
ratios = np.array([r[0] for r in rows])
print(f"greedy cost ratio over all solvable: mean {ratios.mean():.2f}, median {np.median(ratios):.2f}, "
      f"max {ratios.max():.2f}, share >=1.5x: {(ratios>=1.5).mean():.0%}, share >=2x: {(ratios>=2).mean():.0%}")
# most extreme instance where A* also saves >= 40% of UCS expansions
pick = next(r for r in rows if r[5] <= 0.6 * r[4])
ratio, seed, s, g, ue, ae, ge, uc, gc = pick
print("showcase:", dict(seed=seed, start=s, goal=g, greedy_ratio=round(ratio, 2), ucs_exp=ue,
                         astar_exp=ae, greedy_exp=ge, opt_cost=round(uc, 1), greedy_cost=round(gc, 1)))
rank = [r[1] for r in rows].index(seed) + 1
print(f"showcase rank by greedy ratio: {rank} of {len(rows)}")
U = generate_universe(N, seed, hazard=HZ)
res = {a: run_algo(a, U, s, g, record_explored=True) for a in ("ucs", "greedy", "astar", "bfs")}
for a, r in res.items():
    print(a, round(r.cost, 1), r.expanded, r.hops)
save_comparison(U, res, s, g, "results/showcase_greedy_trap.png",
                f"Greedy trap  |  seed {seed}, {N} bodies, uniform hazard, {int(U.blocked.sum())} obstacles")
json.dump(dict(seed=seed, n=N, start=s, goal=g, scanned=SCAN, solvable=len(rows), rank=rank,
               median_ratio=float(np.median(ratios)), mean_ratio=float(ratios.mean()),
               res={a: dict(cost=r.cost, expanded=r.expanded, hops=r.hops) for a, r in res.items()}),
          open("results/showcase.json", "w"), indent=1)
