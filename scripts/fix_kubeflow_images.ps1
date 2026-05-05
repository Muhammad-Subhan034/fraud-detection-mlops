$ErrorActionPreference = "Stop"

Write-Host "Patching Kubeflow Pipelines images from gcr.io to ghcr.io..."

# UI (frontend)
kubectl -n kubeflow set image deployment/ml-pipeline-ui `
  ml-pipeline-ui=ghcr.io/kubeflow/kfp-frontend:master

# API server
kubectl -n kubeflow set image deployment/ml-pipeline `
  ml-pipeline-api-server=ghcr.io/kubeflow/kfp-api-server:master

Write-Host "Restarting deployments..."
kubectl -n kubeflow rollout restart deployment/ml-pipeline-ui
kubectl -n kubeflow rollout restart deployment/ml-pipeline

Write-Host "Done. Re-check with:"
Write-Host "kubectl get pods -n kubeflow"
