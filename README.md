# MLOps Batch Job

A minimal MLOps-style batch pipeline that loads OHLCV data, computes a rolling mean on close price, generates a binary signal, and outputs structured metrics.

## Local Run

### Install dependencies
pip install -r requirements.txt

### Run the job
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log

## Docker

### Build
docker build -t mlops-task .

### Run
docker run --rm mlops-task

## Example metrics.json
{
  "version": "v1",
  "status": "success",
  "seed": 42,
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 38
}