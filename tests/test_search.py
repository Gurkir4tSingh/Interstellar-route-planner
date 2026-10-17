"""Correctness tests. networkx is used ONLY here, as an independent oracle.

Run from the project folder:  python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search import run_algo, best_first  # noqa: E402
from universe import Universe, generate_universe, pick_start_goal  # noqa: E402


def to_nx(U: Universe) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(U.n))
    for i, nb in enumerate(U.adj):
        for j, c in nb:
            G.add_edge(i, j, weight=c)
    return G


def manual_universe(coords, edges) -> Universe:
    """Hand-built universe. edges = [(a, b, hazard)]; cost = euclid * hazard."""
    coords = np.array(coords, dtype=float)
    adj = [[] for _ in coords]
    for a, b, hz in edges:
        c = float(np.linalg.norm(coords[a] - coords[b]) * hz)
        adj[a].append((b, c))
        adj[b].append((a, c))
    return Universe(seed=-1, coords=coords, adj=adj, blocked=np.zeros(len(coords), bool))


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        a, b = generate_universe(200, 7), generate_universe(200, 7)
        self.assertTrue(np.array_equal(a.coords, b.coords))
        self.assertEqual(a.adj, b.adj)
        self.assertTrue(np.array_equal(a.blocked, b.blocked))

    def test_different_seeds_differ(self):
        self.assertFalse(np.array_equal(generate_universe(200, 1).coords,
                                        generate_universe(200, 2).coords))

    def test_hazard_at_least_one(self):
        for mode in ("constant", "uniform", "exponential", "heavy"):
            U = generate_universe(300, 3, hazard=mode)
            for i, nb in enumerate(U.adj):
                for j, c in nb:
                    self.assertGreaterEqual(c, U.euclid(i, j) - 1e-9)

    def test_obstacles_removed_from_graph(self):
        U = generate_universe(400, 5, obstacle_frac=0.3)
        self.assertEqual(int(U.blocked.sum()), 120)
        for i in np.flatnonzero(U.blocked):
            self.assertEqual(U.adj[i], [])
        for i, nb in enumerate(U.adj):
            for j, _ in nb:
                self.assertFalse(U.blocked[j])

    def test_start_goal_not_blocked(self):
        U = generate_universe(300, 9, obstacle_frac=0.4)
        for d in ("any", "near", "mid", "far"):
            s, g = pick_start_goal(U, 9, d)
            self.assertFalse(U.blocked[s] or U.blocked[g])
            self.assertNotEqual(s, g)


class TestAgainstOracle(unittest.TestCase):
    def test_ucs_and_astar_match_dijkstra(self):
        checked = 0
        for seed in range(40):
            for n in (60, 300, 1000):
                U = generate_universe(n, seed)
                s, g = pick_start_goal(U, seed)
                G = to_nx(U)
                if not nx.has_path(G, s, g):
                    continue
                opt = nx.dijkstra_path_length(G, s, g)
                u = run_algo("ucs", U, s, g)
                a = run_algo("astar", U, s, g)
                self.assertAlmostEqual(u.cost, opt, places=6)
                self.assertAlmostEqual(a.cost, opt, places=6)
                checked += 1
        self.assertGreater(checked, 50)

    def test_bfs_fewest_hops(self):
        for seed in range(30):
            U = generate_universe(500, seed)
            s, g = pick_start_goal(U, seed)
            G = to_nx(U)
            if not nx.has_path(G, s, g):
                continue
            r = run_algo("bfs", U, s, g)
            self.assertEqual(r.hops, nx.shortest_path_length(G, s, g))

    def test_greedy_never_beats_optimal(self):
        for seed in range(40):
            U = generate_universe(500, seed)
            s, g = pick_start_goal(U, seed)
            u, gr, bf = (run_algo(x, U, s, g) for x in ("ucs", "greedy", "bfs"))
            if u.found:
                self.assertGreaterEqual(gr.cost, u.cost - 1e-9)
                self.assertGreaterEqual(bf.cost, u.cost - 1e-9)

    def test_paths_are_valid(self):
        for seed in range(20):
            U = generate_universe(400, seed, obstacle_frac=0.2)
            s, g = pick_start_goal(U, seed)
            for name in ("ucs", "greedy", "astar", "bfs"):
                r = run_algo(name, U, s, g)
                if not r.found:
                    continue
                self.assertEqual(r.path[0], s)
                self.assertEqual(r.path[-1], g)
                self.assertAlmostEqual(U.path_cost(r.path), r.cost, places=6)  # edges exist
                self.assertFalse(any(U.blocked[v] for v in r.path))

    def test_heuristic_is_admissible_and_consistent(self):
        U = generate_universe(500, 11, hazard="heavy")
        s, g = pick_start_goal(U, 11)
        from search import heuristic_table
        h = heuristic_table(U, g)
        G = to_nx(U)
        dist_to_goal = nx.single_source_dijkstra_path_length(G, g)
        for v, d in dist_to_goal.items():
            self.assertLessEqual(h[v], d + 1e-9)           # admissible
        for i, nb in enumerate(U.adj):
            for j, c in nb:
                self.assertLessEqual(h[i], c + h[j] + 1e-9)  # consistent

    def test_astar_never_expands_more_than_ucs(self):
        for seed in range(40):
            U = generate_universe(800, seed)
            s, g = pick_start_goal(U, seed)
            u, a = run_algo("ucs", U, s, g), run_algo("astar", U, s, g)
            if u.found:
                self.assertLessEqual(a.expanded, u.expanded)

    def test_astar_h0_equals_ucs(self):
        U = generate_universe(500, 4)
        s, g = pick_start_goal(U, 4)
        u, a0 = run_algo("ucs", U, s, g), run_algo("astar_h0", U, s, g)
        self.assertAlmostEqual(u.cost, a0.cost, places=9)
        self.assertEqual(u.expanded, a0.expanded)


class TestEdgeCases(unittest.TestCase):
    def test_start_equals_goal(self):
        U = generate_universe(100, 1)
        s, _ = pick_start_goal(U, 1)
        for name in ("ucs", "greedy", "astar", "bfs"):
            r = run_algo(name, U, s, s)
            self.assertTrue(r.found)
            self.assertEqual(r.cost, 0.0)
            self.assertEqual(r.path, [s])

    def test_unreachable_goal_reported_not_found(self):
        U = manual_universe([(0, 0, 0), (10, 0, 0), (500, 500, 500)], [(0, 1, 1.0)])
        for name in ("ucs", "greedy", "astar", "bfs"):
            r = run_algo(name, U, 0, 2)
            self.assertFalse(r.found)

    def test_fully_blocked_universe(self):
        U = generate_universe(80, 2, obstacle_frac=0.95)
        s, g = pick_start_goal(U, 2)
        found = {a: run_algo(a, U, s, g).found for a in ("ucs", "greedy", "astar", "bfs")}
        self.assertEqual(len(set(found.values())), 1)   # all four agree on solvability


class TestGreedyTrap(unittest.TestCase):
    """Hand-built trap: Greedy is lured toward the goal by A and pays for it."""

    def setUp(self):
        #           S         A          X            B          C         G
        coords = [(0, 0, 0), (70, 0, 0), (80, 60, 0), (40, 40, 0), (80, 40, 0), (100, 0, 0)]
        edges = [(0, 1, 1.0),   # S-A  cheap, and A looks great (h=30)
                 (1, 2, 5.0),   # A-X  very hazardous
                 (2, 5, 1.0),   # X-G
                 (0, 3, 1.0),   # S-B  looks bad (h=72) but opens the real route
                 (3, 4, 1.0),   # B-C
                 (4, 5, 1.0)]   # C-G
        self.U = manual_universe(coords, edges)

    def test_trap(self):
        u = run_algo("ucs", self.U, 0, 5)
        a = run_algo("astar", self.U, 0, 5)
        gr = run_algo("greedy", self.U, 0, 5)
        self.assertEqual(u.path, [0, 3, 4, 5])
        self.assertEqual(a.path, [0, 3, 4, 5])
        self.assertEqual(gr.path, [0, 1, 2, 5])       # lured into the hazardous detour
        self.assertAlmostEqual(u.cost, a.cost, places=9)
        self.assertGreater(gr.cost, 3 * u.cost)       # ~3.1x worse


if __name__ == "__main__":
    unittest.main()
