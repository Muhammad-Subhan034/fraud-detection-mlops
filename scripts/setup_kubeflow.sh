#!/bin/bash
# =============================================================
# Script: setup_kubeflow.sh
# Description: Sets up Minikube + Kubeflow Pipelines environment
# Run this locally before submitting the pipeline
# =============================================================

set -e

echo "============================================="
echo " Fraud Detection MLOps - Environment Setup"
echo "============================================="

# --- Step 1: Start Minikube ---
echo "[1/8] Starting Minikube..."
minikube start \
  --cpus=4 \
  --memory=8192 \
  --disk-size=50g \
  --driver=docker \
  --kubernetes-version=v1.27.0

echo "Minikube started successfully."

# --- Step 2: Enable addons ---
echo "[2/8] Enabling Minikube addons..."
minikube addons enable ingress
minikube addons enable storage-provisioner
minikube addons enable metrics-server

# --- Step 3: Install Kubeflow Pipelines ---
echo "[3/8] Installing Kubeflow Pipelines (standalone)..."
export PIPELINE_VERSION=2.0.5

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

echo "Waiting for Kubeflow Pipelines pods to be ready..."
kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s
kubectl wait --for=condition=ready pod -l app=ml-pipeline-ui -n kubeflow --timeout=300s

# --- Step 4: Create fraud-detection namespace ---
echo "[4/8] Creating fraud-detection namespace..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: fraud-detection
  labels:
    app: fraud-detection
    environment: production
EOF

# --- Step 5: Apply Resource Quotas ---
echo "[5/8] Applying resource quotas..."
kubectl apply -f kubeflow/resource-quota.yaml

# --- Step 6: Create Persistent Volumes ---
echo "[6/8] Creating persistent volumes..."
kubectl apply -f kubeflow/persistent-volumes.yaml

# --- Step 7: Create ServiceAccount & RBAC ---
echo "[7/8] Creating service accounts..."
kubectl apply -f kubeflow/rbac.yaml

# --- Step 8: Port-forward Kubeflow UI ---
echo "[8/8] Setup complete!"
echo ""
echo "============================================="
echo " Access Kubeflow UI:"
echo " Run: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
echo " Then open: http://localhost:8080"
echo "============================================="
