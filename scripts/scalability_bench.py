"""Wall-clock / RSS at 10, 50, and 100 clients.

Same backbone and per-client budget as the 18-run CIFAR grid (MobileNetV3-Small,
ImageNet transfer, 1000 samples/client, CIFAR-10, IID). Only two rounds and no
interruption, so this stays a cost probe rather than repeating the 20-round study.
CIFAR-10 train has 50k images, so 100 clients cannot all receive 1000 samples.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.federated import FederatedSimulation  # noqa: E402

SAMPLES_PER_CLIENT = 1000


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def run_one(n_clients: int) -> dict:
    sim = FederatedSimulation()
    t0 = time.perf_counter()
    sim.start(
        {
            "num_clients": n_clients,
            "rounds": 2,
            "local_epochs": 3,
            "samples_per_client": SAMPLES_PER_CLIENT,
            "batch_size": 64,
            "lr": 0.01,
            "seed": 42,
            "dataset_name": "cifar10",
            "data_distribution": "iid",
            "model_name": "mobilenet_v3_small",
            "transfer_learning": True,
            "interruption_schedule": [],
            "submission_window_s": 0.0,
        }
    )
    sim.wait()
    elapsed = time.perf_counter() - t0
    mean_samples = 0.0
    if sim.clients:
        mean_samples = sum(c.samples for c in sim.clients) / float(len(sim.clients))
    return {
        "num_clients": n_clients,
        "model_name": "mobilenet_v3_small",
        "dataset_name": "cifar10",
        "samples_per_client_requested": SAMPLES_PER_CLIENT,
        "mean_client_samples": round(mean_samples, 1),
        "wall_clock_s": round(elapsed, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "device": str(sim.device),
        "final_val_acc": sim.history[-1]["val_acc"] if sim.history else None,
    }


def main() -> None:
    out_dir = ROOT / "paper" / "SoftwareX" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_one(n) for n in (10, 50, 100)]
    path = out_dir / "scalability.json"
    path.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
