"""Client availability helpers: schedules, empty-round FedAvg, class metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def parse_interruption_schedule(raw: Any) -> List[Dict[str, int]]:
    """Parse JSON-like interruption windows.

    Each item: {"client_id": 3, "offline_from": 8, "offline_to": 12}
    Rounds are inclusive. Invalid entries are skipped.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        import json

        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("interruption_schedule must be a list of windows")

    out: List[Dict[str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item["client_id"])
            lo = int(item["offline_from"])
            hi = int(item["offline_to"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid interruption window: {item}") from exc
        if cid < 0 or lo < 1 or hi < lo:
            raise ValueError(
                f"Invalid interruption window client_id={cid} offline_from={lo} offline_to={hi}"
            )
        out.append({"client_id": cid, "offline_from": lo, "offline_to": hi})
    return out


def scheduled_offline(schedule: Sequence[Dict[str, int]], client_id: int, round_idx: int) -> bool:
    for w in schedule:
        if int(w["client_id"]) == client_id and int(w["offline_from"]) <= round_idx <= int(
            w["offline_to"]
        ):
            return True
    return False


def row_normalize_confusion(matrix: List[List[float]]) -> List[List[float]]:
    norm: List[List[float]] = []
    for row in matrix:
        s = float(sum(row))
        if s <= 0:
            norm.append([0.0 for _ in row])
        else:
            norm.append([float(v) / s for v in row])
    return norm


def class_metrics_from_confusion(
    matrix: List[List[float]],
    labels: Sequence[str],
) -> List[Dict[str, Any]]:
    n = len(matrix)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        tp = float(matrix[i][i]) if i < len(matrix[i]) else 0.0
        fp = sum(float(matrix[r][i]) for r in range(n) if r != i)
        fn = sum(float(matrix[i][c]) for c in range(n) if c != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        name = labels[i] if i < len(labels) else str(i)
        rows.append(
            {
                "class_id": i,
                "class_name": name,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
    return rows
