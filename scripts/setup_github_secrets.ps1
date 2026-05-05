# scripts/setup_github_secrets.ps1
# ============================================================
# Automatically adds all required GitHub Actions secrets to:
#   Muhammad-Subhan034/fraud-detection-mlops
#
# REQUIRES: GitHub CLI (gh) installed and authenticated
#   Install: winget install --id GitHub.cli
#   Auth:    gh auth login
#
# USAGE:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_github_secrets.ps1
# ============================================================

$Repo = "Muhammad-Subhan034/fraud-detection-mlops"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  GitHub Secrets Setup for fraud-detection-mlops" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check gh CLI is installed
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: GitHub CLI (gh) not found." -ForegroundColor Red
    Write-Host "Install it with:  winget install --id GitHub.cli" -ForegroundColor Yellow
    exit 1
}

# Check authenticated
$authStatus = gh auth status 2>&1
if ($authStatus -match "not logged in") {
    Write-Host "ERROR: Not logged into GitHub CLI." -ForegroundColor Red
    Write-Host "Run:  gh auth login" -ForegroundColor Yellow
    exit 1
}

Write-Host "GitHub CLI authenticated. Setting secrets on: $Repo" -ForegroundColor Green
Write-Host ""

# ── Secret 1: KUBEFLOW_HOST ───────────────────────────────────────────────────
# For local Minikube: http://$(minikube ip):8080
# For demo purposes we use localhost — the CI step has continue-on-error
$KubeflowHost = "http://localhost:8080"
Write-Host "Setting KUBEFLOW_HOST = $KubeflowHost ..." -NoNewline
$KubeflowHost | gh secret set KUBEFLOW_HOST --repo $Repo
Write-Host " Done" -ForegroundColor Green

# ── Secret 2: KUBECONFIG ─────────────────────────────────────────────────────
# Minimal kubeconfig pointing to localhost (demo safe — step has continue-on-error)
$DemoKubeconfig = @"
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://127.0.0.1:6443
    insecure-skip-tls-verify: true
  name: fraud-detection-demo
contexts:
- context:
    cluster: fraud-detection-demo
    user: demo-user
    namespace: fraud-detection
  name: fraud-detection-demo
current-context: fraud-detection-demo
users:
- name: demo-user
  user:
    token: demo-token-not-real
"@

Write-Host "Setting KUBECONFIG (demo localhost kubeconfig) ..." -NoNewline
$DemoKubeconfig | gh secret set KUBECONFIG --repo $Repo
Write-Host " Done" -ForegroundColor Green

# ── Secret 3: SLACK_WEBHOOK_URL (optional — set to empty/placeholder) ────────
Write-Host "Setting SLACK_WEBHOOK_URL (placeholder — replace with real URL for Slack alerts) ..." -NoNewline
"https://hooks.slack.com/services/placeholder/not/real" | gh secret set SLACK_WEBHOOK_URL --repo $Repo
Write-Host " Done" -ForegroundColor Green

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  All secrets set! Summary:" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

gh secret list --repo $Repo

Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. The CI pipeline will now run on your next git push." -ForegroundColor White
Write-Host "  2. Stage 1 (lint + tests) and Stage 2 (Docker build) will pass fully." -ForegroundColor White
Write-Host "  3. Stage 3 (Kubeflow) will compile the pipeline YAML and show a" -ForegroundColor White
Write-Host "     simulation summary (kubectl steps skip gracefully with no cluster)." -ForegroundColor White
Write-Host "  4. To trigger Stage 4 manually:" -ForegroundColor White
Write-Host "     gh workflow run fraud-detection-cicd.yml \`" -ForegroundColor Gray
Write-Host "       --repo $Repo \`" -ForegroundColor Gray
Write-Host "       -f trigger_reason=recall_drop \`" -ForegroundColor Gray
Write-Host "       -f current_recall=0.65 \`" -ForegroundColor Gray
Write-Host "       -f drift_score=0.25" -ForegroundColor Gray
Write-Host ""
Write-Host "  5. To force a Prometheus alert for demo:" -ForegroundColor White
Write-Host "     Set FORCE_RECALL=0.60 in your .env, then: docker-compose up" -ForegroundColor Gray
Write-Host ""
