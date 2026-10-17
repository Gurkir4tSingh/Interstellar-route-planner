"""Search algorithms for the Interstellar Route Planner, implemented from scratch.

UCS, Greedy Best-First, A* and BFS all use graph search with a closed set.
No graph/search libraries are used here (only heapq / deque from the standard library
and numpy for the straight-line heuristic).

Metrics recorded per run: found, path cost, hops, nodes expanded, nodes generated,
peak frontier size, runtime.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from universe import Universe

ALGOS = ("ucs", "greedy", "astar", "bfs")
ALGO_LABEL = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*", "bfs": "BFS",
              "astar_h0": "A* (h=0)", "astar_w2": "A* (2x h)"}


@dataclass
class Result:
    algo: str
    found: bool
    cost: float
    hops: int
    expanded: int
    generated: int
    max_frontier: int
    time_ms: float
    path: list = field(default_factory=list)
    explored: set = field(default_factory=set)


def heuristic_table(U: Universe, goal: int, weight: float = 1.0) -> list:
    """h(v) = straight-line 3D distance from v to the goal (times an optional weight)."""
    return (weight * np.linalg.norm(U.coords - U.coords[goal], axis=1)).tolist()


def _build_path(parent: dict, node: int) -> list:
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def best_first(U: Universe, start: int, goal: int, mode: str,
               h_weight: float = 1.0, record_explored: bool = False,
               label: str | None = None) -> Result:
    """Generic best-first graph search.

    mode = 'ucs'    : priority f = g
           'greedy' : priority f = h
           'astar'  : priority f = g + h   (h_weight=0 -> equivalent to UCS, h_weight>1 -> inadmissible)
    """
    t0 = perf_counter()
    adj = U.adj
    h = heuristic_table(U, goal, h_weight) if mode in ("greedy", "astar") else None

    def key(g_val: float, v: int) -> float:
        if mode == "ucs":
            return g_val
        if mode == "greedy":
            return h[v]
        return g_val + h[v]

    tie = itertools.count()
    g_cost = {start: 0.0}
    parent = {start: None}
    frontier = [(key(0.0, start), next(tie), start)]
    closed = set()
    expanded, generated, max_frontier = 0, 1, 1

    found_node = None
    while frontier:
        _, _, u = heapq.heappop(frontier)
        if u in closed:
            continue                      # stale heap entry
        closed.add(u)
        expanded += 1
        if u == goal:
            found_node = u
            break
        gu = g_cost[u]
        for v, c in adj[u]:
            if v in closed:
                continue
            ng = gu + c
            if mode == "greedy":
                if v in g_cost:           # greedy ignores cost: first discovery wins
                    continue
            elif v in g_cost and ng >= g_cost[v]:
                continue
            g_cost[v] = ng
            parent[v] = u
            heapq.heappush(frontier, (key(ng, v), next(tie), v))
            generated += 1
        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    elapsed = (perf_counter() - t0) * 1000.0
    name = label or {"ucs": "ucs", "greedy": "greedy", "astar": "astar"}[mode]
    if found_node is None:
        return Result(name, False, float("inf"), 0, expanded, generated, max_frontier, elapsed,
                      explored=set(closed) if record_explored else set())
    path = _build_path(parent, goal)
    return Result(name, True, g_cost[goal], len(path) - 1, expanded, generated, max_frontier,
                  elapsed, path, set(closed) if record_explored else set())


def bfs(U: Universe, start: int, goal: int, record_explored: bool = False) -> Result:
    """Breadth-first search: fewest hops, ignores edge costs (baseline)."""
    t0 = perf_counter()
    adj = U.adj
    parent = {start: None}
    queue = deque([start])
    expanded, generated, max_frontier = 0, 1, 1
    closed = set()
    found = start == goal
    while queue and not found:
        u = queue.popleft()
        closed.add(u)
        expanded += 1
        for v, _ in adj[u]:
            if v in parent:
                continue
            parent[v] = u
            generated += 1
            if v == goal:                 # goal test at generation
                found = True
                break
            queue.append(v)
        if len(queue) > max_frontier:
            max_frontier = len(queue)
    elapsed = (perf_counter() - t0) * 1000.0
    if not found:
        return Result("bfs", False, float("inf"), 0, expanded, generated, max_frontier, elapsed,
                      explored=closed if record_explored else set())
    path = _build_path(parent, goal)
    return Result("bfs", True, U.path_cost(path), len(path) - 1, expanded, generated,
                  max_frontier, elapsed, path, closed if record_explored else set())


def run_algo(name: str, U: Universe, start: int, goal: int, record_explored: bool = False) -> Result:
    if name == "bfs":
        return bfs(U, start, goal, record_explored)
    if name in ("ucs", "greedy", "astar"):
        return best_first(U, start, goal, name, record_explored=record_explored)
    if name == "astar_h0":
        return best_first(U, start, goal, "astar", h_weight=0.0, label="astar_h0",
                          record_explored=record_explored)
    if name == "astar_w2":
        return best_first(U, start, goal, "astar", h_weight=2.0, label="astar_w2",
                          record_explored=record_explored)
    raise ValueError(f"unknown algorithm: {name}")
