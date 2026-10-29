"""FastAPI inference service for Cloud Run deployment."""
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow, os
import numpy as np
import structlog

app = FastAPI(title="ML Pipeline GCP - Inference")
log = structlog.get_logger()
_model = None


def get_model():
    global _model
    if _model is None:
        uri = os.environ["MLFLOW_MODEL_URI"]
        _model = mlflow.pyfunc.load_model(uri)
        log.info("model_loaded", uri=uri)
    return _model


class PredictRequest(BaseModel):
    features: list[list[float]]


class PredictResponse(BaseModel):
    scores: list[float]
    labels: list[int]
    threshold: float = 0.5


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    model   = get_model()
    X       = np.array(req.features)
    scores  = model.predict(X).tolist()
    threshold = float(os.environ.get("THRESHOLD", "0.5"))
    labels  = [int(s >= threshold) for s in scores]
    log.info("prediction", n=len(scores), positives=sum(labels))
    return PredictResponse(scores=scores, labels=labels, threshold=threshold)


@app.get("/health")
async def health():
    return {"status": "ok"}
