# FLInterrupt

FLInterrupt is an interactive federated learning framework/simulator for studying how client connectivity affects global training behavior.

It combines a FastAPI + PyTorch backend with a React dashboard, so you can:
- configure federated runs,
- interrupt/reconnect clients while training is live,
- track round-by-round global metrics,
- inspect per-client participation and performance,
- run repeated K-fold validation,
- analyze the mean confusion matrix.

FLInterrupt is designed for education, experimentation, and rapid prototyping.

## What FLInterrupt Does

- Simulates synchronous round-based federated training with FedAvg.
- Supports multiple datasets: `cifar10`, `cifar100`, `mnist`, `fashionmnist`.
- Supports IID and Non-IID data partitioning across clients.
- Supports multiple model backbones (ResNet, MobileNet, EfficientNet, DenseNet, ConvNeXt, ViT, SqueezeNet, and a lightweight CNN).
- Lets you manually interrupt and reconnect specific clients during an active run.
- Tracks client connectivity and training impact (participated rounds, missed rounds, streaks, reconnect stats, latest train/val metrics).
- Exposes post-run evaluation with repeated K-fold validation and average confusion matrix.
- Exports figures from the dashboard for reporting.

## Architecture

```text
Frontend (React + Vite, polling every 1s)
        |
        v
Backend API (FastAPI)
        |
        v
Federated Simulation Engine (PyTorch, worker thread)
        |
        v
Datasets + Model Factory + FedAvg Aggregation + Evaluation
```

### Backend
- `backend/app/main.py`
  - REST API endpoints (`/api/start`, `/api/state`, `/api/interrupt`, `/api/reconnect`, `/api/stop`, `/api/cross-validate`)
  - Input validation with Pydantic
  - CORS for local frontend (`localhost:5173`)
- `backend/app/federated.py`
  - Core simulation lifecycle and training loop
  - Client state management and interrupt/reconnect behavior
  - Dataset loading/transforms/partitioning
  - Weighted FedAvg aggregation
  - Repeated K-fold validation + mean confusion matrix
- `backend/app/model.py`
  - Model factory and classifier head replacement for selected architecture

### Frontend
- `frontend/src/App.jsx`
  - Configuration panel and controls
  - Runtime state polling (`/api/state`)
  - Interrupt/reconnect actions
  - Global evolution chart, repeated K-fold chart, confusion matrix
  - Figure export to PNG
- `frontend/src/App.css`
  - Dashboard layout and visual styling

## Prerequisites

Install these on your machine:
- Python `3.9+`
- `pip` (latest recommended)
- Node.js `18+` and `npm`

Optional but useful:
- A GPU-capable PyTorch setup (`MPS` on Apple Silicon or `CUDA` on NVIDIA).  
  If unavailable, FLInterrupt runs on CPU automatically.

Network requirements on first run:
- Dataset download from `torchvision` sources (saved under `backend/data/`)
- Optional pretrained ImageNet weights download when transfer learning is enabled

## Installation and Run

Use two terminals: one for backend, one for frontend.

### 1) Clone and enter project

```bash
git clone https://github.com/tudordavidz/FLInterrupt.git
cd FLInterrupt
```

### 2) Backend setup and run

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Quick health check (optional):

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

FastAPI docs:
- Swagger UI: `http://localhost:8000/docs`

### 3) Frontend setup and run

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:
- `http://localhost:5173`

## Quick Start Workflow

1. Open the dashboard and choose:
   - number of clients/rounds/local epochs,
   - dataset + distribution mode,
   - model architecture + transfer learning.
2. Start simulation.
3. During run, interrupt or reconnect clients from the client cards.
4. Track global metrics and active-client dynamics live.
5. Stop run when needed.
6. Launch repeated K-fold validation and inspect the mean confusion matrix.
7. Export charts for reports.

## API Endpoints

- `GET /health`  
  Basic service status.
- `GET /api/state`  
  Full simulation state (config, round index, logs, history, clients, available options).
- `POST /api/start`  
  Start a new simulation with config.
- `POST /api/stop`  
  Request graceful stop.
- `POST /api/interrupt`  
  Interrupt selected clients (or a random subset by count).
- `POST /api/reconnect`  
  Reconnect selected disconnected clients (or all).
- `POST /api/cross-validate`  
  Run repeated K-fold validation against current global model weights.

## Configuration Highlights

Main training parameters:
- `num_clients`, `rounds`, `local_epochs`
- `samples_per_client`, `batch_size`, `lr`, `seed`
- `dataset_name`, `data_distribution`
- `model_name`, `transfer_learning`

Cross-validation parameters:
- `repeats`, `k_folds`, `max_samples`

## Project Layout

```text
FLInterrupt/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── federated.py
│   │   └── model.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   └── package.json
└── README.md
```

## Troubleshooting

- Frontend says "Cannot reach backend":
  - confirm backend is running on `http://localhost:8000`
  - verify `uvicorn app.main:app --reload --port 8000`
- Port already in use:
  - backend: run on another port and update `API_BASE` in `frontend/src/App.jsx`
  - frontend: Vite will usually suggest another free port automatically
- First run is slow:
  - datasets and optional pretrained weights may be downloading
- Cross-validation fails with config mismatch:
  - run training with the same dataset/model/transfer-learning settings first

## Notes

- The backend auto-selects compute device in this order: `MPS` -> `CUDA` -> `CPU`.
- This project focuses on observability and experimentation in federated learning under connectivity dynamics.
