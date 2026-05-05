Write-Host "============================================="
Write-Host " Fraud Detection MLOps - Windows Setup"
Write-Host "============================================="

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [string]$Command,
        [string]$ErrorMessage
    )
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

Write-Host "[1/8] Checking required tools..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed."
}
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    throw "kubectl is not installed."
}

$minikubeExe = "minikube"
if (-not (Get-Command minikube -ErrorAction SilentlyContinue)) {
    $fallbackMinikube = "C:\Program Files\Kubernetes\Minikube\minikube.exe"
    if (Test-Path $fallbackMinikube) {
        $minikubeExe = "`"$fallbackMinikube`""
    } else {
        throw "minikube is not installed. Install and re-run this script."
    }
}

Write-Host "[2/8] Starting Minikube..."
Invoke-Checked "$minikubeExe start --cpus=4 --memory=5000 --disk-size=50g --driver=docker --kubernetes-version=v1.30.0" "Failed to start Minikube"

Write-Host "[3/8] Enabling addons..."
Invoke-Checked "$minikubeExe addons enable ingress" "Failed to enable ingress addon"
Invoke-Checked "$minikubeExe addons enable storage-provisioner" "Failed to enable storage-provisioner addon"
Invoke-Checked "$minikubeExe addons enable metrics-server" "Failed to enable metrics-server addon"

Write-Host "[4/8] Installing Kubeflow Pipelines..."
$PIPELINE_VERSION = "2.0.5"
Invoke-Checked "kubectl apply -k `"github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION`"" "Failed applying Kubeflow cluster-scoped resources"
Invoke-Checked "kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io" "CRD establishment wait failed"
Invoke-Checked "kubectl apply -k `"github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION`"" "Failed applying Kubeflow platform manifests"

Write-Host "[5/8] Creating namespace..."
Invoke-Checked "kubectl create namespace fraud-detection --dry-run=client -o yaml | kubectl apply -f -" "Failed creating fraud-detection namespace"

Write-Host "[6/8] Applying quotas and storage..."
Invoke-Checked "kubectl apply -f `"kubeflow/resource-quota.yaml`"" "Failed applying resource quota"
Invoke-Checked "kubectl apply -f `"kubeflow/persistent-volumes.yaml`"" "Failed applying persistent volumes"

Write-Host "[7/8] Applying RBAC..."
Invoke-Checked "kubectl apply -f `"kubeflow/rbac.yaml`"" "Failed applying RBAC"

Write-Host "[8/8] Setup complete."
Write-Host "Run this in a separate terminal to access Kubeflow UI:"
Write-Host "kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
Write-Host "Kubeflow UI: http://localhost:8080"
