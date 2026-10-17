"""Seeded random universe generator for the Interstellar Route Planner.

A universe is an undirected weighted graph:
  * n celestial bodies placed uniformly at random in a 3D cube
  * each body is linked to its k nearest neighbours (k drawn per body from [k_min, k_max])
  * every link has cost = euclidean_distance * hazard, hazard >= 1
  * a fraction of bodies are obstacles; obstacles (and their links) are removed from the graph

Because cost >= straight-line distance, the 3D euclidean distance to the goal is an
admissible (and consistent) heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HAZARD_MODES = ("constant", "uniform", "exponential", "heavy")


def sample_hazard(rng: np.random.Generator, mode: str, size: int) -> np.ndarray:
    """Hazard multipliers, always >= 1."""
    if mode == "constant":
        return np.ones(size)
    if mode == "uniform":      # 1 .. 3, mean 2
        return 1.0 + rng.uniform(0.0, 2.0, size)
    if mode == "exponential":  # 1 + Exp(1), mean 2, unbounded tail
        return 1.0 + rng.exponential(1.0, size)
    if mode == "heavy":        # 1 + Pareto(1.5): rare, very dangerous routes
        return 1.0 + rng.pareto(1.5, size)
    raise ValueError(f"unknown hazard mode: {mode}")


@dataclass
class Universe:
    seed: int
    coords: np.ndarray                 # (n, 3)
    adj: list                          # adj[i] = [(j, cost), ...]   (obstacle-free graph)
    blocked: np.ndarray                # (n,) bool, True = obstacle
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.coords)

    @property
    def edge_count(self) -> int:
        return sum(len(a) for a in self.adj) // 2

    def euclid(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def edge_cost(self, i: int, j: int):
        for v, c in self.adj[i]:
            if v == j:
                return c
        return None

    def path_cost(self, path) -> float:
        """Recompute a path's cost from the graph itself (independent of any search)."""
        total = 0.0
        for a, b in zip(path, path[1:]):
            c = self.edge_cost(a, b)
            if c is None:
                raise ValueError(f"no edge {a}->{b}")
            total += c
        return total


def generate_universe(
    n: int,
    seed: int,
    k_min: int = 2,
    k_max: int = 6,
    hazard: str = "uniform",
    obstacle_frac: float = 0.10,
    size: float = 1000.0,
) -> Universe:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, size, (n, 3))
    k_each = rng.integers(k_min, k_max + 1, n)
    k_cap = min(k_max, n - 1)

    # k nearest neighbours via chunked brute-force distances (fine up to ~5000 bodies)
    nbrs = np.empty((n, k_cap), dtype=np.int64)
    sq = (coords ** 2).sum(axis=1)
    for lo in range(0, n, 500):
        hi = min(lo + 500, n)
        d2 = sq[lo:hi, None] + sq[None, :] - 2.0 * coords[lo:hi] @ coords.T
        d2[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
        part = np.argpartition(d2, k_cap - 1, axis=1)[:, :k_cap]
        order = np.take_along_axis(d2, part, axis=1).argsort(axis=1)
        nbrs[lo:hi] = np.take_along_axis(part, order, axis=1)

    edges = set()
    for i in range(n):
        for j in nbrs[i, : min(int(k_each[i]), k_cap)]:
            j = int(j)
            edges.add((i, j) if i < j else (j, i))
    edge_list = sorted(edges)

    lengths = np.array([np.linalg.norm(coords[a] - coords[b]) for a, b in edge_list])
    costs = lengths * sample_hazard(rng, hazard, len(edge_list))

    n_block = int(round(obstacle_frac * n))
    blocked = np.zeros(n, dtype=bool)
    if n_block:
        blocked[rng.choice(n, n_block, replace=False)] = True

    adj = [[] for _ in range(n)]
    for (a, b), c in zip(edge_list, costs):
        if blocked[a] or blocked[b]:
            continue
        adj[a].append((b, float(c)))
        adj[b].append((a, float(c)))

    params = dict(n=n, k_min=k_min, k_max=k_max, hazard=hazard,
                  obstacle_frac=obstacle_frac, size=size)
    return Universe(seed=seed, coords=coords, adj=adj, blocked=blocked, params=params)


def pick_start_goal(U: Universe, seed: int, distance: str = "any"):
    """Pick (start, goal) among non-obstacle bodies.

    distance: 'any' (uniform random pair), or 'near' / 'mid' / 'far', meaning the goal
    sits in the 5-20th / 45-55th / 85-100th percentile of straight-line distance from start.
    """
    rng = np.random.default_rng(10_000 + seed)
    free = np.flatnonzero(~U.blocked)
    s = int(rng.choice(free))
    cand = free[free != s]
    if distance == "any":
        return s, int(rng.choice(cand))
    d = np.linalg.norm(U.coords[cand] - U.coords[s], axis=1)
    order = cand[np.argsort(d)]
    lo, hi = {"near": (0.05, 0.20), "mid": (0.45, 0.55), "far": (0.85, 1.00)}[distance]
    band = order[int(lo * len(order)): max(int(hi * len(order)), int(lo * len(order)) + 1)]
    return s, int(rng.choice(band))
