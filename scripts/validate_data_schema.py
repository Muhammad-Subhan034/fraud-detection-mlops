"""
Compatibility wrapper for CI schema validation command.
Delegates to src.utils.validate_data_schema.
"""

import argparse
import logging
import os
import sys

import pandas as pd

from src.utils.validate_data_schema import (
    load_schema,
    validate_against_schema
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-path", required=True)
    parser.add_argument("--sample-data", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.sample_data):
        logger.warning("Sample data not found at %s - skipping validation", args.sample_data)
        sys.exit(0)

    schema = load_schema(args.schema_path)
    df = pd.read_parquet(args.sample_data)
    errors = validate_against_schema(df, schema)

    if errors:
        logger.error("Schema validation FAILED: %s errors", len(errors))
        for err in errors:
            logger.error("  - %s", err)
        sys.exit(1)

    logger.info("Schema validation PASSED")
