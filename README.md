# IEEE CIS Fraud Detection System

## Full MLOps Pipeline with Kubeflow, CI/CD, Monitoring & Explainability

---

## Project Structure

```
fraud-detection/
├── kubeflow/               # Kubeflow pipeline definitions
│   ├── components/         # Individual pipeline components
│   └── pipeline/           # Pipeline assembly + submission
├── src/
│   ├── data/               # Data ingestion, validation, preprocessing
│   ├── models/             # Training, evaluation, cost-sensitive learning
│   ├── api/                # FastAPI inference service
│   ├── monitoring/         # Drift detection, metrics
│   └── explainability/     # SHAP analysis
├── ci_cd/                  # GitHub Actions + Jenkins configs
├── monitoring/             # Prometheus + Grafana configs
├── docker/                 # Dockerfiles
├── tests/                  # Unit + integration tests
└── scripts/                # Utility scripts
```

---

## Quick Start

### Prerequisites
- Minikube or Kubernetes cluster
- kubectl + kfp (Kubeflow Pipelines SDK)
- Docker
- Python 3.9+

### 1. Setup Kubeflow (Minikube)
```bash
bash scripts/setup_kubeflow.sh
```
On Windows PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_kubeflow_windows.ps1
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data
Place CSVs in `data/raw/`

### 4. Submit Kubeflow Pipeline
```bash
python kubeflow/pipeline/submit_pipeline.py
```

### 5. Start Inference API
```bash
docker-compose up -d
```

### 6. Access Dashboards
- Kubeflow UI: http://localhost:8080
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- API Docs: http://localhost:8000/docs

---

## Tasks Overview

| Task | Description | Status |
|------|-------------|--------|
| 1 | Kubeflow Environment + Pipeline | ✅ |
| 2 | Data Challenges (Imbalance, Encoding) | ✅ |
| 3 | Model Complexity (XGBoost, LightGBM, Hybrid) | ✅ |
| 4 | Cost-Sensitive Learning | ✅ |
| 5 | CI/CD Pipeline (GitHub Actions) | ✅ |
| 6 | Observability (Prometheus + Grafana) | ✅ |
| 7 | Drift Simulation | ✅ |
| 8 | Intelligent Retraining | ✅ |
| 9 | Explainability (SHAP) | ✅ |
