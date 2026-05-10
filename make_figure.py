"""Generates a summary figure comparing this solver to the published baselines."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# Numbers from Table 1 of Koblischke et al. 2025
BASELINES = {
    "full-obs": {
        "PhD ref":       100.0,
        "o4-mini-high":   74.0,
        "Claude 3.5 S":   39.5,
        "GPT-4o":         36.1,
        "GPT-4o-mini":    26.7,
    },
    "budget-obs-100": {
        "PhD ref":        82.5,
        "o4-mini-high":   49.0,
        "Claude 3.5 S":   21.5,
        "GPT-4o":         15.5,
        "GPT-4o-mini":     8.3,
    },
}

OURS = {"full-obs": 100.0, "budget-obs-100": 86.4}


def main() -> None:
    full_obs_path = HERE / "results" / "all206_results.json"
    bud_path      = HERE / "results" / "all206_budget100_results.json"
    f_full = json.loads(full_obs_path.read_text())
    f_bud  = json.loads(bud_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mode, this_score in zip(
        axes,
        ["full-obs", "budget-obs-100"],
        [OURS["full-obs"], OURS["budget-obs-100"]],
    ):
        labels = ["This solver"] + list(BASELINES[mode].keys())
        scores = [this_score] + list(BASELINES[mode].values())
        colors = ["#1f77b4"] + (["#2ca02c"] + ["#7f7f7f"] * (len(labels) - 2))
        bars = ax.barh(labels, scores, color=colors)
        for bar, s in zip(bars, scores):
            ax.text(s + 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{s:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Tasks solved (%)")
        ax.set_title(f"{mode}  ({len(f_full['results'])} tasks)")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "Gravity-Bench-v1: independent reference solver vs. published baselines",
        fontsize=12,
    )
    fig.tight_layout()
    out = HERE / "figures" / "comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")

    # Error histogram on full-obs
    errs = [
        r["err_pct"]
        for r in f_full["results"]
        if r["err_pct"] is not None and r["err_pct"] > 0
    ]
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log10(np.array(errs) + 1e-6), bins=30, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("log10(relative error in %)")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Error distribution on full-obs (numeric tasks only)")
    ax.grid(True, alpha=0.3)
    out2 = HERE / "figures" / "error_distribution.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
