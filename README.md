# Production ML Pipeline — GCP

End-to-end production ML pipeline on Google Cloud Platform.
Vertex AI managed training, Cloud Run inference, BigQuery feature engineering, MLflow tracking.

## Architecture

```
BigQuery (raw data)
       ↓
Feature engineering (BigQuery SQL + pandas)
       ↓
Vertex AI Training Job
       ↓
MLflow Model Registry
       ↓
Cloud Run inference service (FastAPI + Docker)
       ↓
Monitoring: drift detection, performance dashboards
```

## Features

- BigQuery-based feature engineering with SQL and pandas
- Vertex AI custom training jobs with GPU support
- Docker-containerised FastAPI inference on Cloud Run
- MLflow experiment tracking and model registry
- CI/CD with GitHub Actions — automated evaluation gates before promotion
- Statistical drift monitoring on features and model outputs
- Automated retraining triggers on drift detection

## Stack

| Component | Technology |
|-----------|-----------|
| Feature engineering | BigQuery, pandas |
| Training | Vertex AI, PyTorch, scikit-learn |
| Registry | MLflow |
| Serving | Cloud Run, FastAPI, Docker |
| CI/CD | GitHub Actions |
| Monitoring | Cloud Monitoring, custom drift |

## Project Structure

```
├── src/
│   ├── features/       # BigQuery feature engineering
│   ├── training/       # Model training scripts
│   ├── evaluation/     # Evaluation harness
│   ├── serving/        # FastAPI inference service
│   └── monitoring/     # Drift detection and alerting
├── configs/
├── tests/
├── .github/workflows/
├── Dockerfile
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/sibi-ai/production-ml-pipeline-gcp
cd production-ml-pipeline-gcp
pip install -r requirements.txt
# Configure GCP credentials
gcloud auth application-default login
python src/features/build_features.py --config configs/pipeline.yaml
python src/training/train.py --config configs/pipeline.yaml
```
