"""Visualisation: 2D projection (x, y) of the 3D universe with explored nodes and the found path."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from search import ALGO_LABEL  # noqa: E402

BG, EDGE, FREE, OBST = "#0b1020", "#2a3555", "#5b6a94", "#c0504d"
EXPLORED, PATH, TERM = "#e8b04a", "#4fd1c5", "#ffffff"


def draw_map(ax, U, result, start, goal, title=None):
    ax.set_facecolor(BG)
    xy = U.coords[:, :2]
    segs = [(xy[i], xy[j]) for i, nb in enumerate(U.adj) for j, _ in nb if i < j]
    ax.add_collection(LineCollection(segs, colors=EDGE, linewidths=0.4, zorder=1))
    free = ~U.blocked
    ax.scatter(xy[free, 0], xy[free, 1], s=4, c=FREE, zorder=2, linewidths=0)
    ax.scatter(xy[U.blocked, 0], xy[U.blocked, 1], s=14, c=OBST, marker="x",
               linewidths=0.8, zorder=2)
    if result is not None and result.explored:
        ex = sorted(result.explored)
        ax.scatter(xy[ex, 0], xy[ex, 1], s=9, c=EXPLORED, zorder=3, linewidths=0)
    if result is not None and result.found:
        p = result.path
        ax.plot(xy[p, 0], xy[p, 1], color=PATH, lw=2.2, zorder=4)
    ax.scatter(*xy[start], s=90, c=TERM, edgecolors=PATH, linewidths=2, zorder=5)
    ax.scatter(*xy[goal], s=140, c=TERM, marker="*", edgecolors=PATH, linewidths=1.2, zorder=5)
    ax.set_xlim(-20, U.params.get("size", 1000) + 20)
    ax.set_ylim(-20, U.params.get("size", 1000) + 20)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, color="#e6e9f2", fontsize=11, pad=6)


def save_comparison(U, results: dict, start, goal, out_path, suptitle=None):
    """results: {algo_name: Result with explored recorded}. One panel per algorithm."""
    names = list(results)
    cols = 2 if len(names) > 1 else 1
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.6 * rows), facecolor=BG)
    axes = [axes] if len(names) == 1 else list(axes.ravel())
    for ax, name in zip(axes, names):
        r = results[name]
        label = ALGO_LABEL.get(name, name)
        if r.found:
            t = f"{label}  |  cost {r.cost:,.0f}  |  expanded {r.expanded}  |  {r.hops} hops"
        else:
            t = f"{label}  |  NO ROUTE  |  expanded {r.expanded}"
        draw_map(ax, U, r, start, goal, t)
    for ax in axes[len(names):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, color="#e6e9f2", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=BG)
    plt.close(fig)
