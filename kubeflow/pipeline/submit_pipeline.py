"""
Main Kubeflow Pipeline: IEEE CIS Fraud Detection
Assembles all 7 components with retry logic and conditional deployment.
"""

import kfp
from kfp import dsl
from kfp.dsl import (
    component, pipeline, Input, Output,
    Dataset, Model, Metrics, Artifact,
    ContainerSpec
)
from typing import NamedTuple
import os

# ---- Container image (build with Dockerfile.pipeline) ----
BASE_IMAGE = "fraud-detection/pipeline:latest"
NAMESPACE   = "fraud-detection"


# ============================================================
# KFP COMPONENT WRAPPERS
# ============================================================

@component(base_image=BASE_IMAGE, packages_to_install=[])
def data_ingestion_op(
    data_dir: str,
    output_data: Output[Dataset],
    output_stats: Output[Artifact],
) -> None:
    import subprocess, sys
    result = subprocess.run([
        sys.executable, "/app/components/data_ingestion/component.py",
        "--data-dir",    data_dir,
        "--output-path", output_data.path,
        "--stats-path",  output_stats.path,
    ], check=True)


@component(base_image=BASE_IMAGE)
def data_validation_op(
    input_data:        Input[Dataset],
    validation_report: Output[Artifact],
    validation_status: Output[Artifact],
) -> None:
    import subprocess, sys
    subprocess.run([
        sys.executable, "/app/components/data_validation/component.py",
        "--input-path",             input_data.path,
        "--output-report-path",     validation_report.path,
        "--validation-status-path", validation_status.path,
    ], check=True)


@component(base_image=BASE_IMAGE)
def data_preprocessing_op(
    input_data:    Input[Dataset],
    output_dir:    Output[Dataset],
    encoders:      Output[Model],
    stats:         Output[Artifact],
) -> None:
    import subprocess, sys, os
    os.makedirs(output_dir.path, exist_ok=True)
    subprocess.run([
        sys.executable, "/app/components/data_preprocessing/component.py",
        "--input-path",   input_data.path,
        "--output-dir",   output_dir.path,
        "--encoder-path", encoders.path,
        "--stats-path",   stats.path,
    ], check=True)


@component(base_image=BASE_IMAGE)
def feature_engineering_op(
    data_dir:         Input[Dataset],
    output_dir:       Output[Dataset],
    feature_metadata: Output[Artifact],
) -> None:
    import subprocess, sys, os
    os.makedirs(output_dir.path, exist_ok=True)
    subprocess.run([
        sys.executable, "/app/components/feature_engineering/component.py",
        "--data-dir",          data_dir.path,
        "--output-dir",        output_dir.path,
        "--feature-meta-path", feature_metadata.path,
    ], check=True)


@component(base_image=BASE_IMAGE)
def model_training_op(
    data_dir:       Input[Dataset],
    model_dir:      Output[Model],
    metrics:        Output[Metrics],
    mlflow_uri:     str = "http://mlflow-service:5000",
) -> None:
    import subprocess, sys, os
    os.makedirs(model_dir.path, exist_ok=True)
    subprocess.run([
        sys.executable, "/app/components/model_training/component.py",
        "--data-dir",     data_dir.path,
        "--output-dir",   model_dir.path,
        "--metrics-path", metrics.path,
        "--mlflow-uri",   mlflow_uri,
    ], check=True)


@component(base_image=BASE_IMAGE)
def model_evaluation_op(
    model_dir:        Input[Model],
    data_dir:         Input[Dataset],
    eval_output_dir:  Output[Dataset],
    eval_metrics:     Output[Metrics],
    deploy_decision:  Output[Artifact],
) -> None:
    import subprocess, sys, os
    os.makedirs(eval_output_dir.path, exist_ok=True)
    subprocess.run([
        sys.executable, "/app/components/model_evaluation/component.py",
        "--model-dir",           model_dir.path,
        "--data-dir",            data_dir.path,
        "--output-dir",          eval_output_dir.path,
        "--eval-metrics-path",   eval_metrics.path,
        "--deploy-decision-path", deploy_decision.path,
    ], check=True)


@component(base_image=BASE_IMAGE)
def conditional_deployment_op(
    model_dir:         Input[Model],
    eval_metrics:      Input[Metrics],
    deploy_decision:   Input[Artifact],
    deployment_status: Output[Artifact],
) -> None:
    import subprocess, sys
    subprocess.run([
        sys.executable, "/app/components/deployment/component.py",
        "--model-dir",              model_dir.path,
        "--eval-metrics-path",      eval_metrics.path,
        "--deploy-decision-path",   deploy_decision.path,
        "--deployment-status-path", deployment_status.path,
    ], check=True)


# ============================================================
# PIPELINE DEFINITION
# ============================================================

@pipeline(
    name="fraud-detection-pipeline",
    description="IEEE CIS Fraud Detection: Full MLOps pipeline with conditional deployment",
)
def fraud_detection_pipeline(
    data_dir:   str = "/mnt/data/raw",
    mlflow_uri: str = "http://mlflow-service.fraud-detection.svc.cluster.local:5000",
):
    # ---- Step 1: Data Ingestion ----
    ingestion = data_ingestion_op(data_dir=data_dir)
    ingestion.set_retry(num_retries=3, backoff_duration="60s")
    ingestion.set_cpu_request("500m").set_memory_request("1Gi")
    ingestion.set_cpu_limit("2").set_memory_limit("4Gi")

    # ---- Step 2: Data Validation ----
    validation = data_validation_op(
        input_data=ingestion.outputs["output_data"]
    )
    validation.set_retry(num_retries=2, backoff_duration="30s")
    validation.set_cpu_request("250m").set_memory_request("512Mi")
    validation.after(ingestion)

    # ---- Step 3: Data Preprocessing ----
    preprocessing = data_preprocessing_op(
        input_data=ingestion.outputs["output_data"]
    )
    preprocessing.set_retry(num_retries=2, backoff_duration="30s")
    preprocessing.set_cpu_request("1").set_memory_request("4Gi")
    preprocessing.set_cpu_limit("4").set_memory_limit("8Gi")
    preprocessing.after(validation)

    # ---- Step 4: Feature Engineering ----
    feat_eng = feature_engineering_op(
        data_dir=preprocessing.outputs["output_dir"]
    )
    feat_eng.set_retry(num_retries=2, backoff_duration="30s")
    feat_eng.set_cpu_request("1").set_memory_request("4Gi")
    feat_eng.after(preprocessing)

    # ---- Step 5: Model Training ----
    training = model_training_op(
        data_dir=feat_eng.outputs["output_dir"],
        mlflow_uri=mlflow_uri,
    )
    training.set_retry(num_retries=1, backoff_duration="120s")
    training.set_cpu_request("2").set_memory_request("8Gi")
    training.set_cpu_limit("4").set_memory_limit("16Gi")
    training.after(feat_eng)

    # ---- Step 6: Model Evaluation ----
    evaluation = model_evaluation_op(
        model_dir=training.outputs["model_dir"],
        data_dir=feat_eng.outputs["output_dir"],
    )
    evaluation.set_retry(num_retries=2, backoff_duration="30s")
    evaluation.set_cpu_request("1").set_memory_request("4Gi")
    evaluation.after(training)

    # ---- Step 7: Conditional Deployment ----
    deployment = conditional_deployment_op(
        model_dir=training.outputs["model_dir"],
        eval_metrics=evaluation.outputs["eval_metrics"],
        deploy_decision=evaluation.outputs["deploy_decision"],
    )
    deployment.set_retry(num_retries=2, backoff_duration="60s")
    deployment.set_cpu_request("250m").set_memory_request("256Mi")
    deployment.after(evaluation)


# ============================================================
# PIPELINE SUBMISSION
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host",      default="http://localhost:8080", help="Kubeflow UI URL")
    parser.add_argument("--data-dir",  default="/mnt/data/raw")
    parser.add_argument("--mlflow",    default="http://mlflow-service.fraud-detection.svc.cluster.local:5000")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    # Compile pipeline to YAML
    from kfp import compiler
    pipeline_yaml = "fraud_detection_pipeline.yaml"
    compiler.Compiler().compile(fraud_detection_pipeline, pipeline_yaml)
    print(f"Pipeline compiled to: {pipeline_yaml}")

    if not args.compile_only:
        # Submit to Kubeflow
        client = kfp.Client(host=args.host)
        run = client.create_run_from_pipeline_func(
            fraud_detection_pipeline,
            arguments={
                "data_dir":   args.data_dir,
                "mlflow_uri": args.mlflow,
            },
            run_name="fraud-detection-run-v1",
            namespace=NAMESPACE,
            enable_caching=True,
        )
        print(f"Pipeline submitted! Run ID: {run.run_id}")
        print(f"Track at: {args.host}/#/runs/details/{run.run_id}")
