"""
scripts/wait_for_pipeline.py
Polls Kubeflow pipeline run status until completion or timeout.
Used in CI/CD Stage 3 to block deployment until training finishes.
"""

import argparse
import sys
import time
import logging
import kfp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TERMINAL_STATES = {"Succeeded", "Failed", "Error", "Skipped"}


def wait_for_latest_run(host: str, experiment_name: str = "fraud-detection",
                         timeout: int = 3600, poll_interval: int = 30) -> bool:
    client = kfp.Client(host=host)

    # Get the most recent run for the experiment
    experiment = client.get_experiment(experiment_name=experiment_name)
    runs = client.list_runs(experiment_id=experiment.experiment_id, page_size=1,
                             sort_by="created_at desc")
    if not runs.runs:
        logger.error("No pipeline runs found")
        return False

    run = runs.runs[0]
    run_id = run.run_id
    run_name = run.display_name
    logger.info(f"Monitoring run: {run_name} (ID: {run_id})")

    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            logger.error(f"Timeout after {timeout}s waiting for pipeline run")
            return False

        run_detail = client.get_run(run_id=run_id)
        state = run_detail.state
        logger.info(f"Run state: {state} (elapsed: {elapsed:.0f}s)")

        if state in TERMINAL_STATES:
            if state == "Succeeded":
                logger.info("✅ Pipeline run SUCCEEDED")
                return True
            else:
                logger.error(f"❌ Pipeline run {state}")
                return False

        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",           required=True)
    parser.add_argument("--timeout",        type=int, default=3600)
    parser.add_argument("--poll-interval",  type=int, default=30)
    parser.add_argument("--experiment",     default="fraud-detection")
    args = parser.parse_args()

    success = wait_for_latest_run(
        host=args.host,
        experiment_name=args.experiment,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )
    sys.exit(0 if success else 1)
