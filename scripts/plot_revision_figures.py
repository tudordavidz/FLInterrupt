"""Plot all 18 CIFAR runs on one axes (each seed as its own line)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "paper" / "SoftwareX" / "experiments"
OUT = ROOT / "paper" / "SoftwareX"

NAME_LABEL = {
    "control": "Connected",
    "moderate": "Moderate",
    "severe": "Severe",
}
DIST_LABEL = {"iid": "IID", "noniid": "non-IID"}
COLORS = {
    ("control", "iid"): "#08519c",
    ("control", "noniid"): "#6baed6",
    ("moderate", "iid"): "#d94801",
    ("moderate", "noniid"): "#fdae6b",
    ("severe", "iid"): "#a50f15",
    ("severe", "noniid"): "#fb6a4a",
}
SEED_STYLE = {
    42: dict(ls="-", marker="o"),
    7: dict(ls="--", marker="s"),
    123: dict(ls=":", marker="^"),
}


def load_runs():
    runs = []
    if not EXP.exists():
        return runs
    order = ("control", "moderate", "severe")
    dist_order = ("iid", "noniid")
    seed_order = (42, 7, 123)
    by_tag = {}
    for path in sorted(EXP.glob("*_seed*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text())
        by_tag[(data["name"], data["distribution"], int(data["seed"]))] = data
    for name in order:
        for dist in dist_order:
            for seed in seed_order:
                key = (name, dist, seed)
                if key in by_tag:
                    runs.append(by_tag[key])
    return runs


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip figure generation", file=sys.stderr)
        sys.exit(0)

    runs = load_runs()
    if not runs:
        print("No experiment JSON yet; skip plots", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    for data in runs:
        name = data["name"]
        dist = data["distribution"]
        seed = int(data["seed"])
        history = data["history"]
        rounds = [h["round"] for h in history]
        acc = [h["val_acc"] for h in history]
        style = SEED_STYLE[seed]
        ax.plot(
            rounds,
            acc,
            color=COLORS[(name, dist)],
            ls=style["ls"],
            marker=style["marker"],
            markevery=1,
            markersize=5,
            lw=1.8,
            label=f"{NAME_LABEL[name]} · {DIST_LABEL[dist]} · seed {seed}",
        )

    ax.set_xlabel("Federated round")
    ax.set_ylabel("Global accuracy")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(list(range(1, 21)))
    ax.set_yticks([i / 10 for i in range(0, 11)])
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9, labelleft=True)
    ax.grid(True, alpha=0.35)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    for stem in (OUT / "global_accuracy_18_runs", OUT / "Fig3"):
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=170)
        print("Wrote", stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
