"""Result charts for the progress deck (dark surface, validated 4-slot categorical palette)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

S = pd.read_csv("results/summary.csv")

SURF, GRID, AXIS, INK, INK2, MUTED = "#0b1020", "#1c2442", "#2e3858", "#e6e9f2", "#aab3cc", "#8b95b3"
COL = {"astar": "#3987e5", "greedy": "#d95926", "ucs": "#199e70", "bfs": "#c98500"}   # slots 1-4, fixed order
NAME = {"astar": "A*", "greedy": "Greedy", "ucs": "UCS", "bfs": "BFS"}
plt.rcParams.update({"font.size": 17, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.edgecolor": AXIS, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)


def get(exp, cond, algo, col):
    r = S[(S.experiment == exp) & (S.condition == str(cond)) & (S.algo == algo)]
    return float(r[col].iloc[0])


# ---- Chart 1: search effort vs map size (log-log) -------------------------------------
sizes = [50, 100, 250, 500, 1000, 2500, 5000]
fig, ax = plt.subplots(figsize=(10, 6.4), dpi=100)
for a in ("ucs", "bfs", "astar", "greedy"):
    y = [get("size", n, a, "mean_expanded") for n in sizes]
    ax.plot(sizes, y, color=COL[a], lw=2, solid_capstyle="round", label=NAME[a], zorder=3)
    ax.scatter(sizes, y, s=64, color=COL[a], edgecolors=SURF, linewidths=2, zorder=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(sizes)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_yticks([10, 100, 1000])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_ylim(3, 3500)
ax.set_xlim(40, 7500)
ax.set_xlabel("Celestial bodies in the map", labelpad=10)
ax.set_ylabel("Nodes expanded (mean of 30 seeds)", labelpad=10)
clean(ax)
ax.text(5000 * 1.12, get("size", 5000, "astar", "mean_expanded"), "A*", color=INK, va="center", fontsize=17)
ax.text(5000 * 1.12, get("size", 5000, "greedy", "mean_expanded"), "Greedy", color=INK, va="center", fontsize=17)
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=17, labelcolor=INK2, handlelength=1.6,
          columnspacing=1.4, borderaxespad=0.2)
fig.subplots_adjust(left=0.12, right=0.93, top=0.97, bottom=0.14)
fig.savefig("results/chart_effort_vs_size.png")
plt.close(fig)

# ---- Chart 2: extra cost over optimal, by hazard distribution ----------------------------
hz = ["constant", "uniform", "exponential", "heavy"]
fig, ax = plt.subplots(figsize=(10, 6.4), dpi=100)
x = np.arange(len(hz))
w = 0.26
vals = {}
for i, a in enumerate(("greedy", "bfs")):
    v = [100 * (get("hazard", h, a, "mean_cost_ratio") - 1) for h in hz]
    vals[a] = v
    pos = x + (i - 0.5) * (w + 0.03)
    ax.bar(pos, v, width=w, color=COL[a], label=NAME[a], zorder=3)
    for p, val in zip(pos, v):
        ax.text(p, val + 2.5, f"+{val:.0f}%", ha="center", va="bottom", color=INK, fontsize=16)
ax.axhline(0, color=AXIS, lw=1.5, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([h.capitalize() for h in hz])
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"+{int(v)}%" if v else "0"))
ax.set_xlabel("Hazard distribution (n = 1,000)", labelpad=10)
ax.set_ylabel("Extra path cost vs optimal", labelpad=10)
clean(ax)
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=17, labelcolor=INK2, handlelength=1.2,
          borderaxespad=0.2)
fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.14)
fig.savefig("results/chart_greedy_cost.png")
plt.close(fig)

# ---- Chart 3: heuristic quality (two panels) -----------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 6.0), dpi=100, gridspec_kw={"wspace": 0.38})
pct = [get("hazard", h, "astar", "mean_expanded_pct_of_ucs") for h in hz]
a1.bar(range(4), pct, width=0.55, color=COL["astar"], zorder=3)
for i, v in enumerate(pct):
    a1.text(i, v + 2.5, f"{v:.0f}%", ha="center", color=INK, fontsize=16)
a1.set_xticks(range(4))
a1.set_xticklabels(["const.", "unif.", "expo.", "heavy"], fontsize=15)
a1.set_ylim(0, 112)
a1.set_yticks([0, 25, 50, 75, 100])
a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
a1.set_title("By hazard distribution", color=INK2, fontsize=17, pad=12, loc="left")
a1.set_ylabel("A* nodes as % of UCS", labelpad=10)
clean(a1)

var = [("astar_h0", "h = 0"), ("astar", "h = d"), ("astar_w2", "h = 2d")]
pct2 = [get("heuristic", "n=1000", a, "mean_expanded_pct_of_ucs") for a, _ in var]
a2.bar(range(3), pct2, width=0.55, color=COL["astar"], zorder=3)
for i, v in enumerate(pct2):
    a2.text(i, v + 2.5, f"{v:.0f}%", ha="center", color=INK, fontsize=16)
a2.set_xticks(range(3))
a2.set_xticklabels([n for _, n in var], fontsize=15)
a2.set_ylim(0, 112)
a2.set_yticks([0, 25, 50, 75, 100])
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
a2.set_title("By heuristic (uniform)", color=INK2, fontsize=17, pad=12, loc="left")
clean(a2)
fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.12)
fig.savefig("results/chart_heuristic.png")
plt.close(fig)

# ---- numbers quoted in the deck (printed so the slide text can be checked against them) ----
print("chart1 mean expanded:")
for a in ("ucs", "bfs", "astar", "greedy"):
    print(" ", a, [round(get("size", n, a, "mean_expanded"), 1) for n in sizes])
print("A* % of UCS by size:", [round(get("size", n, "astar", "mean_expanded_pct_of_ucs"), 1) for n in sizes])
print("Greedy % of UCS by size:", [round(get("size", n, "greedy", "mean_expanded_pct_of_ucs"), 1) for n in sizes])
print("greedy extra % by hazard:", [round(v, 1) for v in vals["greedy"]], " bfs:", [round(v, 1) for v in vals["bfs"]])
print("A* % of UCS by hazard:", [round(v, 1) for v in pct])
for n in ("n=1000", "n=5000"):
    for a, nm in var:
        print("heuristic", n, nm, "exp_pct %.1f" % get("heuristic", n, a, "mean_expanded_pct_of_ucs"),
              "mean cost ratio %.4f" % get("heuristic", n, a, "mean_cost_ratio"),
              "max %.3f" % get("heuristic", n, a, "max_cost_ratio"),
              "optimal_frac %.2f" % get("heuristic", n, a, "optimal_frac"))
print("greedy cost ratio n=1000/5000: %.3f %.3f" % (get("size", 1000, "greedy", "mean_cost_ratio"),
                                                   get("size", 5000, "greedy", "mean_cost_ratio")))
print("greedy max ratio n=1000/5000: %.2f %.2f" % (get("size", 1000, "greedy", "max_cost_ratio"),
                                                  get("size", 5000, "greedy", "max_cost_ratio")))
print("greedy optimal_frac n=50..5000:", [round(get("size", n, "greedy", "optimal_frac"), 2) for n in sizes])
print("bfs cost ratio n=1000:", round(get("size", 1000, "bfs", "mean_cost_ratio"), 3))
print("time ms n=5000 ucs/astar/greedy/bfs:", [round(get("size", 5000, a, "mean_time_ms"), 2) for a in ("ucs", "astar", "greedy", "bfs")])
