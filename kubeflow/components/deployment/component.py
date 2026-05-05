"""
Component 7: Conditional Deployment
Deploys model to Kubernetes if evaluation metrics meet thresholds.
Rolls back automatically if deployment fails health checks.
"""

import argparse
import json
import logging
import subprocess
import time
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HEALTH_CHECK_URL = "http://fraud-api-service.fraud-detection.svc.cluster.local:8000/health"
HEALTH_CHECK_RETRIES = 10
HEALTH_CHECK_DELAY = 15  # seconds


def check_deploy_decision(deploy_decision_path: str) -> bool:
    with open(deploy_decision_path) as f:
        decision = f.read().strip()
    logger.info(f"Deploy decision: {decision}")
    return decision == "DEPLOY"


def get_model_version(eval_metrics_path: str) -> str:
    with open(eval_metrics_path) as f:
        metrics = json.load(f)
    auc = metrics.get("auc_roc", 0)
    return f"auc{auc:.4f}".replace(".", "")


def write_deployment_yaml(model_dir: str, model_version: str, output_path: str) -> None:
    yaml_content = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-api
  namespace: fraud-detection
  labels:
    app: fraud-api
    version: "{model_version}"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fraud-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: fraud-api
        version: "{model_version}"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: fraud-detection-sa
      containers:
        - name: fraud-api
          image: fraud-detection/inference-api:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: /models/best_model.joblib
            - name: MODEL_VERSION
              value: "{model_version}"
            - name: LOG_LEVEL
              value: "INFO"
          volumeMounts:
            - name: model-storage
              mountPath: /models
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 5
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: fraud-artifacts-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-api-service
  namespace: fraud-detection
  labels:
    app: fraud-api
spec:
  selector:
    app: fraud-api
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-api-hpa
  namespace: fraud-detection
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
"""
    with open(output_path, "w") as f:
        f.write(yaml_content)
    logger.info(f"Wrote deployment YAML to {output_path}")


def deploy_model(model_dir: str, eval_metrics_path: str,
                 deploy_decision_path: str, deployment_status_path: str) -> None:

    should_deploy = check_deploy_decision(deploy_decision_path)

    if not should_deploy:
        logger.warning("Deployment skipped: evaluation thresholds not met.")
        with open(deployment_status_path, "w") as f:
            json.dump({"status": "SKIPPED", "reason": "metrics_below_threshold"}, f)
        return

    model_version = get_model_version(eval_metrics_path)
    logger.info(f"Deploying model version: {model_version}")

    # Write Kubernetes manifests
    deploy_yaml = "/tmp/fraud-deployment.yaml"
    write_deployment_yaml(model_dir, model_version, deploy_yaml)

    # Apply
    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", deploy_yaml],
            capture_output=True, text=True, timeout=120
        )
        logger.info(f"kubectl apply stdout: {result.stdout}")
        if result.returncode != 0:
            raise RuntimeError(f"kubectl apply failed: {result.stderr}")

        # Wait for rollout
        logger.info("Waiting for rollout to complete...")
        rollout = subprocess.run(
            ["kubectl", "rollout", "status", "deployment/fraud-api",
             "-n", "fraud-detection", "--timeout=300s"],
            capture_output=True, text=True, timeout=360
        )
        if rollout.returncode != 0:
            raise RuntimeError(f"Rollout failed: {rollout.stderr}")

        # Health checks
        logger.info("Running health checks...")
        for i in range(HEALTH_CHECK_RETRIES):
            try:
                resp = requests.get(HEALTH_CHECK_URL, timeout=5)
                if resp.status_code == 200:
                    logger.info("Health check passed ✓")
                    break
            except Exception:
                pass
            logger.info(f"Health check attempt {i+1}/{HEALTH_CHECK_RETRIES}...")
            time.sleep(HEALTH_CHECK_DELAY)
        else:
            raise RuntimeError("Health checks failed after all retries")

        status = {"status": "SUCCESS", "model_version": model_version}
        logger.info(f"✅ Deployment successful: {model_version}")

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        # Rollback
        subprocess.run(["kubectl", "rollout", "undo", "deployment/fraud-api",
                        "-n", "fraud-detection"], timeout=60)
        status = {"status": "FAILED", "error": str(e)}

    with open(deployment_status_path, "w") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--eval-metrics-path", type=str, required=True)
    parser.add_argument("--deploy-decision-path", type=str, required=True)
    parser.add_argument("--deployment-status-path", type=str, required=True)
    args = parser.parse_args()

    deploy_model(args.model_dir, args.eval_metrics_path,
                 args.deploy_decision_path, args.deployment_status_path)
