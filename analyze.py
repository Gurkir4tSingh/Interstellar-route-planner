"""Aggregate raw_results.csv into per-condition summaries (printed + results/summary.json)."""
import json
import sys

import numpy as np
import pandas as pd

ORDER = {"size": ["50", "100", "250", "500", "1000", "2500", "5000"],
         "density": ["k=2", "k=3", "k=4", "k=6"],
         "hazard": ["constant", "uniform", "exponential", "heavy"],
         "obstacles": ["0%", "10%", "20%", "30%", "40%"],
         "distance": ["near", "mid", "far"],
         "heuristic": ["n=1000", "n=5000"]}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ucs = df[df.algo == "ucs"].set_index(["experiment", "condition", "seed"])
    df = df.join(ucs[["expanded", "time_ms"]].rename(columns={"expanded": "ucs_expanded",
                                                              "time_ms": "ucs_time"}),
                 on=["experiment", "condition", "seed"])
    df["cost_ratio"] = df.cost / df.optimal_cost
    df["opt_hit"] = np.isclose(df.cost, df.optimal_cost, rtol=1e-9)
    df["exp_vs_ucs"] = df.expanded / df.ucs_expanded
    rows = []
    for (exp, cond, algo), g in df.groupby(["experiment", "condition", "algo"], sort=False):
        solv = g[g.solvable == 1]
        rows.append(dict(
            experiment=exp, condition=cond, algo=algo, instances=len(g),
            success_rate=g.found.mean(), solvable_rate=g.solvable.mean(),
            mean_cost_ratio=solv.cost_ratio.mean(), max_cost_ratio=solv.cost_ratio.max(),
            optimal_frac=solv.opt_hit.mean(),
            mean_expanded=solv.expanded.mean(), median_expanded=solv.expanded.median(),
            mean_expanded_pct_of_ucs=100 * solv.exp_vs_ucs.mean(),
            mean_generated=solv.generated.mean(), mean_max_frontier=solv.max_frontier.mean(),
            mean_time_ms=solv.time_ms.mean(), mean_hops=solv.hops.mean(),
            mean_straight=solv.straight_dist.mean()))
    return pd.DataFrame(rows)


def main(path="results/raw_results.csv", out="results/summary.json"):
    df = pd.read_csv(path)
    S = summarize(df)
    S.to_json(out, orient="records", indent=1)
    S.to_csv(out.replace(".json", ".csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format)
    for exp, conds in ORDER.items():
        print(f"\n=== {exp} ===")
        sub = S[S.experiment == exp]
        for c in conds:
            print(f"-- {c}")
            print(sub[sub.condition == c][["algo", "solvable_rate", "success_rate", "mean_cost_ratio",
                                           "max_cost_ratio", "optimal_frac", "mean_expanded",
                                           "mean_expanded_pct_of_ucs", "mean_time_ms"]]
                  .to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
