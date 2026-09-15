"""Paper protocol: control / moderate / severe × IID / non-IID × seeds (1000 samples/client)."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.federated import FederatedSimulation  # noqa: E402

MODERATE = [
    {"client_id": 3, "offline_from": 8, "offline_to": 12},
    {"client_id": 4, "offline_from": 8, "offline_to": 12},
    {"client_id": 5, "offline_from": 8, "offline_to": 12},
]
SEVERE = [
    {"client_id": i, "offline_from": 8, "offline_to": 15} for i in range(7)
]

BASE = {
    "num_clients": 10,
    "rounds": 20,
    "local_epochs": 3,
    "samples_per_client": 1000,
    "batch_size": 64,
    "lr": 0.01,
    "dataset_name": "cifar10",
    "model_name": "mobilenet_v3_small",
    "transfer_learning": True,
    "submission_window_s": 0.0,
}


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def run_job(name: str, distribution: str, seed: int, schedule: list) -> dict:
    sim = FederatedSimulation()
    t0 = time.perf_counter()
    sim.start(
        {
            **BASE,
            "seed": seed,
            "data_distribution": distribution,
            "interruption_schedule": schedule,
        }
    )
    sim.wait()
    elapsed = time.perf_counter() - t0
    payload = {
        "name": name,
        "distribution": distribution,
        "seed": seed,
        "wall_clock_s": elapsed,
        "peak_rss_mb": peak_rss_mb(),
        "device": str(sim.device),
        "export": sim.export_experiment(),
        "history": sim.history,
    }
    return payload


def main() -> None:
    out_dir = ROOT / "paper" / "SoftwareX" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dist in ("iid", "noniid"):
        for seed in (42, 7, 123):
            jobs.append(("control", dist, seed, []))
            jobs.append(("moderate", dist, seed, MODERATE))
            jobs.append(("severe", dist, seed, SEVERE))

    summary = []
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = []
    done = {(row["name"], row["distribution"], row["seed"]) for row in summary}

    for name, dist, seed, sched in jobs:
        tag = f"{name}_{dist}_seed{seed}"
        path = out_dir / f"{tag}.json"
        if (name, dist, seed) in done and path.exists():
            print(f"=== skip {tag} (already in summary) ===", flush=True)
            continue
        print(f"=== {tag} ===", flush=True)
        result = run_job(name, dist, seed, sched)
        path.write_text(json.dumps(result, indent=2, default=str))
        last = result["history"][-1] if result["history"] else {}
        summary.append(
            {
                "name": name,
                "distribution": dist,
                "seed": seed,
                "final_val_acc": last.get("val_acc"),
                "final_val_loss": last.get("val_loss"),
                "wall_clock_s": result["wall_clock_s"],
                "peak_rss_mb": result["peak_rss_mb"],
                "device": result["device"],
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary[-1], indent=2), flush=True)


if __name__ == "__main__":
    main()
