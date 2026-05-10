from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from .federated import FederatedSimulation


class SimulationConfig(BaseModel):
    num_clients: int = Field(5, ge=2, le=100)
    rounds: int = Field(5, ge=1, le=100)
    local_epochs: int = Field(1, ge=1, le=5)
    samples_per_client: int = Field(800, ge=100, le=5000)
    batch_size: int = Field(64, ge=16, le=256)
    lr: float = Field(0.01, gt=0.0, le=1.0)
    seed: int = Field(42, ge=0)
    dataset_name: str = Field("cifar10")
    data_distribution: str = Field("iid")
    model_name: str = Field("resnet18")
    transfer_learning: bool = True


class InterruptRequest(BaseModel):
    client_ids: Optional[List[int]] = None
    count: Optional[int] = Field(None, ge=1)


class ReconnectRequest(BaseModel):
    client_ids: Optional[List[int]] = None
    reconnect_all: bool = False


class CrossValidationRequest(BaseModel):
    repeats: int = Field(2, ge=1, le=20)
    k_folds: int = Field(5, ge=2, le=20)
    max_samples: int = Field(3000, ge=500, le=10000)
    dataset_name: Optional[str] = None
    training_model_name: Optional[str] = None
    transfer_learning: Optional[bool] = None


app = FastAPI(title="Federated Learning Simulation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sim = FederatedSimulation()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
def state() -> dict:
    return sim.get_state()


@app.post("/api/start")
def start(config: SimulationConfig) -> dict[str, str]:
    try:
        sim.start(config.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "simulation started"}


@app.post("/api/stop")
def stop() -> dict[str, str]:
    sim.stop()
    return {"message": "stop requested"}


@app.post("/api/interrupt")
def interrupt(req: InterruptRequest) -> dict:
    ids = sim.interrupt_clients(client_ids=req.client_ids, count=req.count)
    return {"message": "clients interrupted", "client_ids": ids}


@app.post("/api/reconnect")
def reconnect(req: ReconnectRequest) -> dict:
    ids = sim.reconnect_clients(
        client_ids=req.client_ids,
        reconnect_all=req.reconnect_all,
    )
    return {"message": "clients reconnected", "client_ids": ids}


@app.post("/api/cross-validate")
def cross_validate(req: CrossValidationRequest) -> dict:
    if sim.get_state().get("running"):
        raise HTTPException(status_code=409, detail="Stop simulation before cross-validation")

    try:
        sim.validate_cv_matches_training_config(
            dataset_name=req.dataset_name,
            model_name=req.training_model_name,
            transfer_learning=req.transfer_learning,
        )
        result = sim.run_repeated_kfold_validation(
            repeats=req.repeats,
            k_folds=req.k_folds,
            max_samples=req.max_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result
