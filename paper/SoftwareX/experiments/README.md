Outputs from `scripts/run_revision_experiments.py` (CIFAR-10, MobileNetV3-Small, 1000 samples/client, 20 rounds, 3 local epochs). Global round records store validation accuracy/loss only.

Jobs: control / moderate / severe × iid / noniid × seeds 42, 7, 123 (18 runs). Completed jobs are skipped on restart.

Each job writes `<name>_<dist>_seed<seed>.json` and updates `summary.json`.

Plot overlays: `python scripts/plot_revision_figures.py` (requires matplotlib).

Scalability: `python scripts/scalability_bench.py` → `scalability.json` (CIFAR-10, MobileNetV3-Small, 1000 samples/client target, 2 rounds, 10/50/100 clients).
