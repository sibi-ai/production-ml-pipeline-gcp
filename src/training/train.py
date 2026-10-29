"""Model training with MLflow tracking and Vertex AI integration."""
import argparse, yaml
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import structlog

log = structlog.get_logger()


def train(cfg: dict):
    df = pd.read_parquet(cfg.get("input_path", "data/features.parquet"))
    feature_cols = [c for c in df.columns if c not in ("entity_id", "event_date", "label")]
    X, y = df[feature_cols].values, df["label"].values

    mlflow.set_tracking_uri(cfg.get("mlflow_uri", "http://localhost:5000"))
    mlflow.set_experiment(cfg.get("experiment", "production-ml-gcp"))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    with mlflow.start_run(run_name=cfg.get("run_name", "xgb_training")):
        mlflow.log_params(cfg.get("model", {}))
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            model = XGBClassifier(**cfg.get("model", {}), tree_method="hist")
            model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            preds = model.predict_proba(X[va])[:, 1]
            auc   = roc_auc_score(y[va], preds)
            aucs.append(auc)
            log.info("fold_done", fold=fold, auc=round(auc, 4))

        mlflow.log_metric("mean_auc", float(np.mean(aucs)))
        mlflow.sklearn.log_model(model, "model")
        log.info("training_complete", mean_auc=round(float(np.mean(aucs)), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
