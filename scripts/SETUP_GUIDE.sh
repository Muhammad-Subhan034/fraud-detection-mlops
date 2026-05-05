# SETUP GUIDE — Fraud Detection MLOps System
# Do these steps IN ORDER after unzipping the project

# =============================================================
# PREREQUISITES (install these first)
# =============================================================
# - Docker Desktop (with 8GB RAM allocated)
# - Minikube: https://minikube.sigs.k8s.io/docs/start/
# - kubectl: https://kubernetes.io/docs/tasks/tools/
# - Python 3.10+
# - Kaggle account + kaggle CLI

# =============================================================
# STEP 1: Download the Dataset from Kaggle
# =============================================================
mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw/
cd data/raw && unzip ieee-fraud-detection.zip && cd ../..
# You should now have: data/raw/train_transaction.csv, train_identity.csv

# =============================================================
# STEP 2: Install Python Dependencies
# =============================================================
pip install -r requirements.txt

# =============================================================
# STEP 3: Run Local Analysis (no Kubernetes needed)
# This runs all ML tasks locally and saves outputs to outputs/analysis/
# =============================================================
python notebooks/run_full_analysis.py
# Outputs: imbalance_comparison.png, SHAP plots, training_results.json,
#          retraining_comparison.csv

# =============================================================
# STEP 4: Start Monitoring Stack (Docker Compose)
# Runs: Fraud API + Prometheus + Grafana + Alertmanager + MLflow
# =============================================================
# First build the API image:
docker build -f docker/Dockerfile.api -t fraud-detection/inference-api:latest .
docker build -f docker/Dockerfile.pipeline -t fraud-detection/pipeline:latest .

# Start the full stack:
docker-compose up -d

# Access points:
#   Grafana:    http://localhost:3000  (admin / frauddetection)
#   Prometheus: http://localhost:9090
#   MLflow:     http://localhost:5000
#   API Docs:   http://localhost:8000/docs
#   Alertmgr:   http://localhost:9093

# Test the API:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 2500.0, "ProductCD": "W", "card4": "visa"}'

# =============================================================
# STEP 5: Import Grafana Dashboards
# =============================================================
# Dashboards auto-provision from monitoring/grafana/dashboards/
# Just open http://localhost:3000 and navigate to Dashboards > Fraud Detection

# =============================================================
# STEP 6: Set Up Kubeflow (optional — needs Minikube)
# =============================================================
chmod +x scripts/setup_kubeflow.sh
bash scripts/setup_kubeflow.sh

# Port-forward Kubeflow UI:
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &

# Copy your data into the PVC (replace with your cluster method):
kubectl cp data/raw/ fraud-detection/$(kubectl get pods -n fraud-detection -o name | head -1):/mnt/data/raw/

# Submit the pipeline:
python kubeflow/pipeline/submit_pipeline.py \
  --host http://localhost:8080 \
  --data-dir /mnt/data/raw

# =============================================================
# STEP 7: Set Up CI/CD (GitHub Actions)
# =============================================================
# 1. Push this project to a GitHub repository
# 2. Add these GitHub Secrets (Settings > Secrets > Actions):
#    - KUBEFLOW_HOST: your Kubeflow URL
#    - KUBECONFIG: base64-encoded kubeconfig file
#    - SLACK_WEBHOOK_URL: for deployment notifications (optional)
#    - GITHUB_TOKEN: auto-provided by GitHub Actions
# 3. Copy ci_cd/fraud-detection-cicd.yml to .github/workflows/
cp ci_cd/fraud-detection-cicd.yml .github/workflows/
# 4. Push to main — pipeline will trigger automatically

# =============================================================
# STEP 8: Enable Intelligent Alerting
# =============================================================
# Start the alert webhook proxy (bridges Alertmanager → GitHub Actions):
export GITHUB_TOKEN=your_token_here
export GITHUB_REPO=your_org/fraud-detection
python scripts/alert_webhook_proxy.py
# Runs on port 5001

# In alertmanager.yml, point the webhook to:
# http://your-proxy-host:5001/alert

# =============================================================
# FILE STRUCTURE SUMMARY
# =============================================================
# fraud-detection/
# ├── kubeflow/
# │   ├── components/
# │   │   ├── data_ingestion/component.py      Task 1 - Step 1
# │   │   ├── data_validation/component.py     Task 1 - Step 2
# │   │   ├── data_preprocessing/component.py  Tasks 1+2 (missing, encoding)
# │   │   ├── feature_engineering/component.py Task 1 - Step 4
# │   │   ├── model_training/component.py      Tasks 2+3+4 (models, imbalance, cost)
# │   │   ├── model_evaluation/component.py    Task 3 (metrics, business impact)
# │   │   └── deployment/component.py          Task 1 (conditional deploy)
# │   ├── pipeline/submit_pipeline.py          Task 1 (pipeline assembly, retry)
# │   ├── resource-quota.yaml                  Task 1 (CPU/memory limits)
# │   ├── persistent-volumes.yaml              Task 1 (artifact storage)
# │   └── rbac.yaml                            Task 1 (namespace isolation)
# ├── src/
# │   ├── api/api.py                           Task 6 (Prometheus metrics)
# │   ├── explainability/shap_analysis.py      Task 9 (SHAP)
# │   └── monitoring/
# │       ├── drift_detection.py               Tasks 7+8 (drift, retraining)
# │       └── metrics_exporter.py              Task 6 (PSI, recall metrics)
# ├── ci_cd/fraud-detection-cicd.yml           Task 5 (GitHub Actions)
# ├── monitoring/
# │   ├── prometheus/prometheus.yml            Task 6 (scrape config)
# │   ├── prometheus/alertmanager.yml          Task 6 (routing + CI/CD trigger)
# │   ├── prometheus/rules/                    Task 6 (alert rules)
# │   └── grafana/dashboards/                  Task 6 (3 dashboards)
# ├── tests/test_all.py                        Task 5 (unit tests)
# ├── notebooks/run_full_analysis.py           All tasks (local demo)
# ├── reports/research_report.md              Research report
# ├── docker-compose.yml                       Local stack
# └── requirements.txt
