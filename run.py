import argparse
import json
import logging
import time
import numpy as np
import pandas as pd
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps batch job")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_fields = ["seed", "window", "version"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: '{field}'")

    return config

def load_data(input_path):
    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Input CSV is empty")

    if "close" not in df.columns:
        raise ValueError("Missing required column: 'close'")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return df

def compute_signals(df, window):
    df["rolling_mean"] = df["close"].rolling(window=window).mean()
    df["signal"] = (df["close"] > df["rolling_mean"]).astype(int)
    # First (window-1) rows will have NaN rolling mean, exclude from signal rate
    valid = df["signal"][window - 1:]
    signal_rate = valid.mean()
    return df, signal_rate

def main():
    args = parse_args()
    setup_logging(args.log_file)
    log = logging.getLogger(__name__)

    start_time = time.time()
    log.info("Job started")

    # We write metrics at the end — prepare a base dict now
    metrics = {"version": "unknown", "status": "error"}

    try:
        # Load and validate config
        log.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        np.random.seed(config["seed"])
        log.info(f"Config loaded — seed={config['seed']}, window={config['window']}, version={config['version']}")
        metrics["version"] = config["version"]
        metrics["seed"] = config["seed"]

        # Load and validate data
        log.info(f"Loading data from: {args.input}")
        df = load_data(args.input)
        log.info(f"Data loaded — {len(df)} rows")

        # Compute rolling mean and signal
        log.info("Computing rolling mean...")
        log.info("Generating signals...")
        df, signal_rate = compute_signals(df, config["window"])

        # Compute timing
        latency_ms = int((time.time() - start_time) * 1000)

        # Build success metrics
        metrics.update({
            "status": "success",
            "rows_processed": len(df),
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
        })

        log.info(f"Rows processed: {len(df)}")
        log.info(f"Signal rate: {round(signal_rate, 4)}")
        log.info(f"Latency: {latency_ms}ms")
        log.info("Job completed successfully")

    except Exception as e:
        log.error(f"Job failed: {e}")
        metrics["error_message"] = str(e)

    finally:
        # Always write metrics, even on failure
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Metrics written to: {args.output}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()