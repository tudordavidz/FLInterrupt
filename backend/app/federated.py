from __future__ import annotations

import copy
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .model import create_model, get_supported_models


SUPPORTED_DATASETS = ["cifar10", "cifar100", "mnist", "fashionmnist"]
SUPPORTED_DISTRIBUTIONS = ["iid", "noniid"]


@dataclass
class ClientState:
    client_id: int
    connected: bool = True
    reconnect_round: int = -1
    samples: int = 0
    last_loss: Optional[float] = None
    interruption_events: int = 0
    reconnect_events: int = 0
    rounds_participated: int = 0
    rounds_missed: int = 0
    disconnect_streak: int = 0
    max_disconnect_streak: int = 0
    last_interrupt_round: Optional[int] = None
    last_reconnect_round: Optional[int] = None
    last_downtime_rounds: int = 0
    metrics_round: Optional[int] = None
    train_acc: Optional[float] = None
    train_loss: Optional[float] = None
    val_acc: Optional[float] = None
    val_loss: Optional[float] = None
    class_distribution: List[Dict[str, Any]] = None


class FederatedSimulation:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.stop_requested = False

        self.config: Dict[str, Any] = {
            "num_clients": 5,
            "rounds": 5,
            "local_epochs": 1,
            "samples_per_client": 800,
            "batch_size": 64,
            "lr": 0.01,
            "seed": 42,
            "dataset_name": "cifar10",
            "data_distribution": "iid",
            "model_name": "resnet18",
            "transfer_learning": True,
        }

        self.current_round = 0
        self.logs: List[str] = []
        self.history: List[Dict[str, Any]] = []
        self.clients: List[ClientState] = []
        self.class_labels: List[str] = [str(i) for i in range(10)]
        self.num_classes = 10

        self.device = self._select_device()
        self.global_model = self._build_model().to(self.device)

    def _select_device(self) -> torch.device:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_model(self) -> nn.Module:
        return create_model(
            model_name=str(self.config.get("model_name", "resnet18")),
            num_classes=self.num_classes,
            transfer_learning=bool(self.config.get("transfer_learning", True)),
        )

    def _build_client_distributions(
        self,
        dataset_obj,
        client_indices: List[List[int]],
    ) -> List[List[Dict[str, Any]]]:
        targets = self._get_targets(dataset_obj)
        out: List[List[Dict[str, Any]]] = []
        for indices in client_indices:
            counts: Dict[int, int] = {}
            for idx in indices:
                label = int(targets[idx])
                counts[label] = counts.get(label, 0) + 1
            summary = [
                {
                    "class_id": cid,
                    "class_name": self.class_labels[cid] if cid < len(self.class_labels) else str(cid),
                    "count": cnt,
                }
                for cid, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            ]
            out.append(summary)
        return out

    def _build_transform(self) -> transforms.Compose:
        dataset_name = str(self.config.get("dataset_name", "cifar10")).lower()
        ops: List[Any] = []
        if dataset_name in {"mnist", "fashionmnist"}:
            ops.append(transforms.Grayscale(num_output_channels=3))
        ops.extend(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        return transforms.Compose(ops)

    def _dataset_is_cached(self, dataset_name: str, root: str = "data") -> bool:
        """Return True if torchvision artifacts for this dataset already exist on disk."""
        base = Path(root)
        d = str(dataset_name).lower()
        if d == "cifar10":
            return (base / "cifar-10-batches-py" / "data_batch_1").is_file()
        if d == "cifar100":
            return (base / "cifar-100-python" / "train").is_file()
        if d == "mnist":
            return (base / "MNIST" / "processed" / "training.pt").is_file()
        if d == "fashionmnist":
            return (base / "FashionMNIST" / "processed" / "training.pt").is_file()
        return False

    def _load_dataset(self, train: bool, transform: transforms.Compose):
        dataset_name = str(self.config.get("dataset_name", "cifar10")).lower()
        if dataset_name == "cifar10":
            return datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
        if dataset_name == "cifar100":
            return datasets.CIFAR100(root="./data", train=train, download=True, transform=transform)
        if dataset_name == "mnist":
            return datasets.MNIST(root="./data", train=train, download=True, transform=transform)
        if dataset_name == "fashionmnist":
            return datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _get_targets(self, dataset_obj) -> List[int]:
        targets = getattr(dataset_obj, "targets", None)
        if targets is None:
            raise ValueError("Selected dataset does not expose targets for partitioning")
        if isinstance(targets, torch.Tensor):
            return [int(v) for v in targets.tolist()]
        return [int(v) for v in targets]

    def _build_client_indices(
        self,
        dataset_obj,
        num_clients: int,
        samples_per_client: int,
        distribution: str,
        seed: int,
    ) -> List[List[int]]:
        total_train = min(len(dataset_obj), num_clients * samples_per_client)
        rng = random.Random(seed)
        all_indices = list(range(len(dataset_obj)))
        rng.shuffle(all_indices)
        selected = all_indices[:total_train]

        # Allocate different sample counts per client while preserving total budget.
        min_per_client = max(10, samples_per_client // 5)
        if min_per_client * num_clients > total_train:
            min_per_client = max(1, total_train // num_clients)

        sizes = [min_per_client for _ in range(num_clients)]
        remaining = total_train - (min_per_client * num_clients)

        weights = [rng.random() + 0.1 for _ in range(num_clients)]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            weight_sum = 1.0

        for i in range(num_clients):
            extra = int(remaining * (weights[i] / weight_sum))
            sizes[i] += extra

        assigned = sum(sizes)
        diff = total_train - assigned
        while diff > 0:
            idx = rng.randrange(num_clients)
            sizes[idx] += 1
            diff -= 1
        while diff < 0:
            idx = rng.randrange(num_clients)
            if sizes[idx] > min_per_client:
                sizes[idx] -= 1
                diff += 1

        if distribution == "noniid":
            targets = self._get_targets(dataset_obj)
            selected.sort(key=lambda idx: targets[idx])
        else:
            rng.shuffle(selected)

        client_indices: List[List[int]] = []
        cursor = 0
        for i in range(num_clients):
            take = sizes[i]
            client_indices.append(selected[cursor : cursor + take])
            cursor += take
        return client_indices

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "config": self.config,
                "device": str(self.device),
                "class_labels": self.class_labels,
                "num_classes": self.num_classes,
                "options": {
                    "datasets": SUPPORTED_DATASETS,
                    "distributions": SUPPORTED_DISTRIBUTIONS,
                    "models": get_supported_models(),
                },
                "current_round": self.current_round,
                "logs": self.logs[-80:],
                "history": self.history,
                "clients": [
                    {
                        "client_id": c.client_id,
                        "connected": c.connected,
                        "reconnect_round": c.reconnect_round,
                        "samples": c.samples,
                        "last_loss": c.last_loss,
                        "interruption_events": c.interruption_events,
                        "reconnect_events": c.reconnect_events,
                        "rounds_participated": c.rounds_participated,
                        "rounds_missed": c.rounds_missed,
                        "disconnect_streak": c.disconnect_streak,
                        "max_disconnect_streak": c.max_disconnect_streak,
                        "last_interrupt_round": c.last_interrupt_round,
                        "last_reconnect_round": c.last_reconnect_round,
                        "last_downtime_rounds": c.last_downtime_rounds,
                        "metrics_round": c.metrics_round,
                        "train_acc": c.train_acc,
                        "train_loss": c.train_loss,
                        "val_acc": c.val_acc,
                        "val_loss": c.val_loss,
                        "class_distribution": c.class_distribution or [],
                    }
                    for c in self.clients
                ],
            }

    def start(self, config: Dict[str, Any]) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("Simulation already running")

            self.config = {
                **self.config,
                **config,
            }
            self.config["dataset_name"] = str(self.config.get("dataset_name", "cifar10")).lower()
            self.config["data_distribution"] = str(self.config.get("data_distribution", "iid")).lower()

            if self.config["dataset_name"] not in SUPPORTED_DATASETS:
                raise RuntimeError(f"Unsupported dataset: {self.config['dataset_name']}")
            if self.config["data_distribution"] not in SUPPORTED_DISTRIBUTIONS:
                raise RuntimeError(f"Unsupported data distribution: {self.config['data_distribution']}")
            self.current_round = 0
            self.logs = []
            self.history = []
            self.stop_requested = False
            self.clients = [
                ClientState(client_id=i)
                for i in range(int(self.config["num_clients"]))
            ]

            if self.config["dataset_name"] == "cifar100":
                self.num_classes = 100
            else:
                self.num_classes = 10

            if self.config["dataset_name"] == "cifar10":
                self.class_labels = [
                    "airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck",
                ]
            elif self.config["dataset_name"] == "cifar100":
                self.class_labels = [str(i) for i in range(100)]
            elif self.config["dataset_name"] == "mnist":
                self.class_labels = [str(i) for i in range(10)]
            else:
                self.class_labels = [
                    "t-shirt", "trouser", "pullover", "dress", "coat",
                    "sandal", "shirt", "sneaker", "bag", "ankle-boot",
                ]

            self.global_model = self._build_model().to(self.device)
            self.running = True

        self._log("Simulation worker starting…")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        with self.lock:
            self.stop_requested = True

    def interrupt_clients(
        self,
        client_ids: Optional[List[int]] = None,
        count: Optional[int] = None,
    ) -> List[int]:
        with self.lock:
            connected = [c for c in self.clients if c.connected]
            if not connected:
                return []

            if client_ids:
                target = [
                    c for c in connected if c.client_id in set(int(i) for i in client_ids)
                ]
            else:
                if count is None:
                    count = 1
                take = max(1, min(int(count), len(connected)))
                target = random.sample(connected, take)

            interrupted_ids: List[int] = []
            for client in target:
                client.connected = False
                # A client interrupted in round r can reconnect from round r+1.
                client.reconnect_round = self.current_round + 1 if self.current_round > 0 else 1
                client.interruption_events += 1
                client.last_interrupt_round = self.current_round if self.current_round > 0 else 0
                interrupted_ids.append(client.client_id)

            if interrupted_ids:
                self.logs.append(
                    f"Manual interruption: clients {interrupted_ids} disconnected"
                )

            return interrupted_ids

    def reconnect_clients(
        self,
        client_ids: Optional[List[int]] = None,
        reconnect_all: bool = False,
    ) -> List[int]:
        with self.lock:
            disconnected = [c for c in self.clients if not c.connected]
            if not disconnected:
                return []

            if reconnect_all:
                target = disconnected
            elif client_ids:
                target = [
                    c
                    for c in disconnected
                    if c.client_id in set(int(i) for i in client_ids)
                ]
            else:
                target = [disconnected[0]]

            reconnected_ids: List[int] = []
            for client in target:
                if (
                    self.running
                    and client.last_interrupt_round is not None
                    and self.current_round <= client.last_interrupt_round
                ):
                    self.logs.append(
                        f"Client {client.client_id} reconnect blocked in round {self.current_round}; "
                        f"available from round {client.last_interrupt_round + 1}"
                    )
                    continue

                client.connected = True
                client.reconnect_round = -1
                client.reconnect_events += 1
                client.disconnect_streak = 0
                client.last_reconnect_round = self.current_round if self.current_round > 0 else 0
                if client.last_interrupt_round is not None and self.current_round > 0:
                    client.last_downtime_rounds = max(
                        0,
                        self.current_round - client.last_interrupt_round,
                    )
                reconnected_ids.append(client.client_id)

            if reconnected_ids:
                self.logs.append(
                    f"Manual reconnect: clients {reconnected_ids} connected"
                )

            return reconnected_ids

    def _log(self, message: str) -> None:
        with self.lock:
            self.logs.append(message)

    def _run(self) -> None:
        try:
            self._train_loop()
        except Exception as exc:
            self._log(f"Simulation failed: {exc}")
        finally:
            with self.lock:
                self.running = False

    def _train_loop(self) -> None:
        cfg = self.config
        self._log("Initializing run (RNG seeds, transforms)…")
        random.seed(int(cfg["seed"]))
        torch.manual_seed(int(cfg["seed"]))

        transform = self._build_transform()

        ds_name = str(cfg.get("dataset_name", "cifar10")).lower()
        needs_download = not self._dataset_is_cached(ds_name)
        if needs_download:
            self._log(
                f"{ds_name}: not found under ./data — downloading from the internet "
                "(first run only; may take a minute)…"
            )
        else:
            self._log(f"{ds_name}: found under ./data — using cached data (no download).")

        self._log(f"Loading training split ({ds_name})…")
        train_data = self._load_dataset(train=True, transform=transform)
        self._log(f"Loading test split ({ds_name})…")
        test_data = self._load_dataset(train=False, transform=transform)
        if needs_download:
            self._log(f"{ds_name}: dataset downloaded and ready under ./data.")
        self._log("Datasets ready; splitting samples across clients…")

        num_clients = int(cfg["num_clients"])
        samples_per_client = int(cfg["samples_per_client"])
        round_count = int(cfg["rounds"])
        local_epochs = int(cfg["local_epochs"])
        batch_size = int(cfg["batch_size"])

        client_indices = self._build_client_indices(
            dataset_obj=train_data,
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            distribution=str(cfg.get("data_distribution", "iid")),
            seed=int(cfg["seed"]),
        )

        for i, c in enumerate(self.clients):
            c.samples = len(client_indices[i])

        client_distributions = self._build_client_distributions(train_data, client_indices)
        for i, c in enumerate(self.clients):
            c.class_distribution = client_distributions[i]

        test_subset = Subset(test_data, list(range(0, min(3000, len(test_data)))))
        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
        train_eval_subset = Subset(train_data, list(range(0, min(3000, len(train_data)))))
        train_eval_loader = DataLoader(train_eval_subset, batch_size=128, shuffle=False)

        self._log(
            f"Simulation started with {num_clients} clients for {round_count} rounds on "
            f"device={self.device}, dataset={cfg.get('dataset_name')}, "
            f"distribution={cfg.get('data_distribution')}, model={cfg.get('model_name')}"
        )

        for round_idx in range(1, round_count + 1):
            should_stop = False
            with self.lock:
                if self.stop_requested:
                    should_stop = True
                self.current_round = round_idx

            if should_stop:
                self._log("Simulation stop requested")
                return

            self._log(f"Round {round_idx}: selecting connected clients")

            active_models: List[Dict[str, torch.Tensor]] = []
            weights: List[int] = []
            contributor_ids: List[int] = []

            trained_updates: List[Dict[str, Any]] = []
            deferred_clients: List[ClientState] = []
            client_by_id = {c.client_id: c for c in self.clients}
            round_submission_map: Dict[int, bool] = {}

            for client in self.clients:
                with self.lock:
                    if self.stop_requested:
                        self._log("Simulation stop requested")
                        return
                    connected_for_training = bool(client.connected)

                if not connected_for_training:
                    self._log(
                        f"Client {client.client_id} disconnected before local training in round {round_idx}; "
                        f"waiting for reconnect window"
                    )
                    deferred_clients.append(client)
                    continue

                subset = Subset(train_data, client_indices[client.client_id])
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
                eval_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
                local_model = copy.deepcopy(self.global_model)
                loss_value, stopped, disconnected_during_train = self._local_train(
                    local_model,
                    loader,
                    local_epochs,
                    float(cfg["lr"]),
                    client.client_id,
                )
                if stopped:
                    self._log("Simulation stop requested")
                    return
                if disconnected_during_train:
                    with self.lock:
                        client.rounds_missed += 1
                        client.disconnect_streak += 1
                        client.max_disconnect_streak = max(
                            client.max_disconnect_streak,
                            client.disconnect_streak,
                        )
                    round_submission_map[client.client_id] = False
                    self._log(
                        f"Client {client.client_id} disconnected during local training in round {round_idx}; "
                        f"update dropped"
                    )
                    continue
                c_train_acc, c_train_loss = self._evaluate(local_model, eval_loader)
                c_val_acc, c_val_loss = self._evaluate(local_model, test_loader)
                trained_updates.append(
                    {
                        "client_id": client.client_id,
                        "samples": client.samples,
                        "state_dict": copy.deepcopy(local_model.state_dict()),
                        "last_loss": loss_value,
                        "train_acc": c_train_acc,
                        "train_loss": c_train_loss,
                        "val_acc": c_val_acc,
                        "val_loss": c_val_loss,
                    }
                )
                self._log(
                    f"Client {client.client_id} finished local training in round {round_idx}; "
                    f"train_acc={c_train_acc:.4f}, train_loss={c_train_loss:.4f}, "
                    f"val_acc={c_val_acc:.4f}, val_loss={c_val_loss:.4f}; "
                    f"waiting for submission phase"
                )

            if deferred_clients:
                self._log(
                    f"Round {round_idx}: reconnect window open for deferred clients (1s)"
                )
                for _ in range(10):
                    with self.lock:
                        if self.stop_requested:
                            self._log("Simulation stop requested")
                            return
                    time.sleep(0.1)

            for client in deferred_clients:
                with self.lock:
                    if self.stop_requested:
                        self._log("Simulation stop requested")
                        return
                    connected_for_training = bool(client.connected)

                if not connected_for_training:
                    with self.lock:
                        client.rounds_missed += 1
                        client.disconnect_streak += 1
                        client.max_disconnect_streak = max(
                            client.max_disconnect_streak,
                            client.disconnect_streak,
                        )
                    round_submission_map[client.client_id] = False
                    self._log(
                        f"Client {client.client_id} still disconnected after reconnect window in round {round_idx}; "
                        f"training skipped"
                    )
                    continue

                subset = Subset(train_data, client_indices[client.client_id])
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
                eval_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
                local_model = copy.deepcopy(self.global_model)
                loss_value, stopped, disconnected_during_train = self._local_train(
                    local_model,
                    loader,
                    local_epochs,
                    float(cfg["lr"]),
                    client.client_id,
                )
                if stopped:
                    self._log("Simulation stop requested")
                    return
                if disconnected_during_train:
                    with self.lock:
                        client.rounds_missed += 1
                        client.disconnect_streak += 1
                        client.max_disconnect_streak = max(
                            client.max_disconnect_streak,
                            client.disconnect_streak,
                        )
                    round_submission_map[client.client_id] = False
                    self._log(
                        f"Client {client.client_id} disconnected during deferred local training in round {round_idx}; "
                        f"update dropped"
                    )
                    continue
                c_train_acc, c_train_loss = self._evaluate(local_model, eval_loader)
                c_val_acc, c_val_loss = self._evaluate(local_model, test_loader)
                trained_updates.append(
                    {
                        "client_id": client.client_id,
                        "samples": client.samples,
                        "state_dict": copy.deepcopy(local_model.state_dict()),
                        "last_loss": loss_value,
                        "train_acc": c_train_acc,
                        "train_loss": c_train_loss,
                        "val_acc": c_val_acc,
                        "val_loss": c_val_loss,
                    }
                )
                self._log(
                    f"Client {client.client_id} finished deferred local training in round {round_idx}; "
                    f"train_acc={c_train_acc:.4f}, train_loss={c_train_loss:.4f}, "
                    f"val_acc={c_val_acc:.4f}, val_loss={c_val_loss:.4f}; "
                    f"waiting for submission phase"
                )

            self._log(
                f"Round {round_idx}: local training finished. Submission window open (1s)"
            )
            for _ in range(10):
                with self.lock:
                    if self.stop_requested:
                        self._log("Simulation stop requested")
                        return
                time.sleep(0.1)

            for update in trained_updates:
                client = client_by_id[int(update["client_id"])]

                with self.lock:
                    client.last_loss = float(update["last_loss"])
                    client.metrics_round = round_idx
                    client.train_acc = float(update["train_acc"])
                    client.train_loss = float(update["train_loss"])
                    client.val_acc = float(update["val_acc"])
                    client.val_loss = float(update["val_loss"])
                    connected_for_submit = bool(client.connected)

                    if connected_for_submit:
                        client.rounds_participated += 1
                        client.disconnect_streak = 0
                    else:
                        client.rounds_missed += 1
                        client.disconnect_streak += 1
                        client.max_disconnect_streak = max(
                            client.max_disconnect_streak,
                            client.disconnect_streak,
                        )
                    round_submission_map[client.client_id] = connected_for_submit

                if connected_for_submit:
                    active_models.append(update["state_dict"])
                    weights.append(int(update["samples"]))
                    contributor_ids.append(client.client_id)
                    self._log(
                        f"Client {client.client_id} submitted round {round_idx}: "
                        f"train_acc={float(update['train_acc']):.4f}, "
                        f"val_acc={float(update['val_acc']):.4f}"
                    )
                else:
                    self._log(
                        f"Client {client.client_id} disconnected at submission; "
                        f"round {round_idx} update dropped"
                    )

            round_submission = [
                {
                    "client_id": cid,
                    "connected": round_submission_map.get(cid, False),
                }
                for cid in sorted(client_by_id.keys())
            ]

            contributor_labels = [f"C{cid}" for cid in sorted(contributor_ids)]
            self._log(
                f"Round {round_idx}: contributors before aggregation: {contributor_labels}"
            )

            if not active_models:
                self._log(
                    f"Round {round_idx}: no active clients; global model unchanged"
                )
                train_acc, train_loss = self._evaluate(self.global_model, train_eval_loader)
                val_acc, val_loss = self._evaluate(self.global_model, test_loader)
            else:
                aggregated = self._fedavg(active_models, weights)
                self.global_model.load_state_dict(aggregated)
                train_acc, train_loss = self._evaluate(self.global_model, train_eval_loader)
                val_acc, val_loss = self._evaluate(self.global_model, test_loader)

            with self.lock:
                self.history.append(
                    {
                        "round": round_idx,
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "accuracy": val_acc,
                        "active_clients": len(active_models),
                        "global_updated": len(active_models) > 0,
                        "client_submission": round_submission,
                    }
                )

            agg_note = (
                "global model after FedAvg"
                if active_models
                else "global model unchanged (no participating clients)"
            )
            self._log(
                f"Round {round_idx} — SERVER AGGREGATE ({agg_note}, "
                f"n={len(active_models)}): "
                f"train_acc={train_acc:.4f}, train_loss={train_loss:.4f}, "
                f"val_acc={val_acc:.4f}, val_loss={val_loss:.4f}"
            )
            time.sleep(0.3)

        self._log("Simulation finished")

    def _local_train(
        self,
        model: nn.Module,
        loader: DataLoader,
        epochs: int,
        lr: float,
        client_id: Optional[int] = None,
    ) -> Tuple[float, bool, bool]:
        model.train()
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            return 0.0, False, False
        optimizer = SGD(trainable_params, lr=lr, momentum=0.9)
        last_loss = 0.0

        for _ in range(epochs):
            for inputs, labels in loader:
                with self.lock:
                    if self.stop_requested:
                        return last_loss, True, False
                    if (
                        client_id is not None
                        and 0 <= client_id < len(self.clients)
                        and not self.clients[client_id].connected
                    ):
                        return last_loss, False, True
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())

        return last_loss, False, False

    def _fedavg(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        weights: List[int],
    ) -> Dict[str, torch.Tensor]:
        total = float(sum(weights))
        aggregated: Dict[str, torch.Tensor] = {}
        first = state_dicts[0]

        for key in first.keys():
            weighted_sum = sum(
                state[key] * (w / total)
                for state, w in zip(state_dicts, weights)
            )
            aggregated[key] = weighted_sum

        return aggregated

    def _evaluate(self, model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += float(loss.item()) * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += int((predicted == labels).sum().item())
                total += int(labels.size(0))

        if total == 0:
            return 0.0, 0.0

        return correct / total, total_loss / total

    def _evaluate_with_confusion(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Tuple[float, float, torch.Tensor]:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += float(loss.item()) * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += int((predicted == labels).sum().item())
                total += int(labels.size(0))

                for true_label, pred_label in zip(labels.cpu(), predicted.cpu()):
                    confusion[int(true_label), int(pred_label)] += 1.0

        if total == 0:
            return 0.0, 0.0, confusion

        return correct / total, total_loss / total, confusion

    def validate_cv_matches_training_config(
        self,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        transfer_learning: Optional[bool] = None,
    ) -> None:
        """Ensure UI matches the config used to build the current global_model weights."""
        if dataset_name is not None:
            if str(dataset_name).lower() != str(self.config.get("dataset_name", "")).lower():
                raise ValueError(
                    "Cross-validation uses the global model from your last simulation run. "
                    f"Server dataset is '{self.config.get('dataset_name')}' but the request had "
                    f"'{dataset_name}'. Start (or finish) a run with the same dataset, or align the form."
                )
        if model_name is not None:
            if str(model_name).lower() != str(self.config.get("model_name", "")).lower():
                raise ValueError(
                    "Cross-validation uses the global model from your last simulation run. "
                    f"Server model is '{self.config.get('model_name')}' but the request had "
                    f"'{model_name}'. Start a run with the same model architecture first."
                )
        if transfer_learning is not None:
            server_tl = bool(self.config.get("transfer_learning", True))
            if bool(transfer_learning) != server_tl:
                raise ValueError(
                    "Cross-validation uses the global model from your last simulation run. "
                    f"Server transfer_learning is {server_tl} but the request had {bool(transfer_learning)}. "
                    "Start a run with the same transfer-learning setting first."
                )

    def run_repeated_kfold_validation(
        self,
        repeats: int,
        k_folds: int,
        max_samples: int,
    ) -> Dict[str, Any]:
        if repeats < 1 or k_folds < 2:
            raise ValueError("repeats must be >= 1 and k_folds must be >= 2")

        transform = self._build_transform()
        ds_key = str(self.config.get("dataset_name", "cifar10")).lower()
        needs_download = not self._dataset_is_cached(ds_key)
        if needs_download:
            self._log(
                f"{ds_key}: not found under ./data — downloading for cross-validation "
                "(first run only; may take a minute)…"
            )
        else:
            self._log(f"{ds_key}: using cached data under ./data for cross-validation.")
        test_data = self._load_dataset(train=False, transform=transform)
        if needs_download:
            self._log(f"{ds_key}: dataset downloaded and ready under ./data.")

        total_available = len(test_data)
        sample_count = min(max_samples, total_available)
        if sample_count < k_folds:
            raise ValueError("k_folds cannot be larger than selected sample count")

        with self.lock:
            eval_model = self._build_model().to(self.device)
            try:
                eval_model.load_state_dict(
                    copy.deepcopy(self.global_model.state_dict()), strict=True
                )
            except Exception as exc:
                raise ValueError(
                    "Cannot load global weights into the evaluation model. "
                    "Train with the current dataset/model settings, then run validation again. "
                    f"Detail: {exc}"
                ) from exc
            seed = int(self.config.get("seed", 42))

        base_indices = list(range(sample_count))
        fold_results: List[Dict[str, Any]] = []
        confusion_sum = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)
        total_folds = repeats * k_folds

        for repeat_idx in range(repeats):
            rng = random.Random(seed + repeat_idx)
            shuffled = base_indices.copy()
            rng.shuffle(shuffled)

            fold_sizes = [sample_count // k_folds] * k_folds
            for i in range(sample_count % k_folds):
                fold_sizes[i] += 1

            start = 0
            for fold_idx, fold_size in enumerate(fold_sizes):
                end = start + fold_size
                val_indices = shuffled[start:end]
                train_indices = shuffled[:start] + shuffled[end:]
                start = end

                train_subset = Subset(test_data, train_indices)
                train_loader = DataLoader(train_subset, batch_size=128, shuffle=False)
                val_subset = Subset(test_data, val_indices)
                val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
                train_acc, train_loss = self._evaluate(eval_model, train_loader)
                val_acc, val_loss, confusion = self._evaluate_with_confusion(eval_model, val_loader)

                fold_results.append(
                    {
                        "repeat": repeat_idx + 1,
                        "fold": fold_idx + 1,
                        "train_accuracy": train_acc,
                        "train_loss": train_loss,
                        "val_accuracy": val_acc,
                        "val_loss": val_loss,
                        # Backward-compatible aliases for older frontends.
                        "accuracy": val_acc,
                        "loss": val_loss,
                    }
                )
                confusion_sum += confusion

        mean_confusion = (confusion_sum / float(total_folds)).tolist()
        mean_train_acc = sum(float(r["train_accuracy"]) for r in fold_results) / float(total_folds)
        mean_train_loss = sum(float(r["train_loss"]) for r in fold_results) / float(total_folds)
        mean_val_acc = sum(float(r["val_accuracy"]) for r in fold_results) / float(total_folds)
        mean_val_loss = sum(float(r["val_loss"]) for r in fold_results) / float(total_folds)

        return {
            "repeats": repeats,
            "k_folds": k_folds,
            "sample_count": sample_count,
            "fold_results": fold_results,
            "mean_train_accuracy": mean_train_acc,
            "mean_train_loss": mean_train_loss,
            "mean_val_accuracy": mean_val_acc,
            "mean_val_loss": mean_val_loss,
            # Backward-compatible aliases for older frontends.
            "mean_accuracy": mean_val_acc,
            "mean_loss": mean_val_loss,
            "mean_confusion_matrix": mean_confusion,
            "labels": self.class_labels,
        }
