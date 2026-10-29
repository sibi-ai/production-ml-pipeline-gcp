"""Feature engineering from BigQuery — builds training and serving feature sets."""
import argparse, yaml
import pandas as pd
from google.cloud import bigquery
import structlog

log = structlog.get_logger()


FEATURE_QUERY = """
SELECT
  entity_id,
  DATE(event_ts) AS event_date,
  COUNT(*)                         AS n_events_7d,
  AVG(value_a)                     AS avg_value_a,
  STDDEV(value_a)                  AS std_value_a,
  MAX(value_b)                     AS max_value_b,
  MIN(value_b)                     AS min_value_b,
  SUM(CASE WHEN flag = 1 THEN 1 ELSE 0 END) AS flag_count,
  label
FROM `{project}.{dataset}.events`
WHERE event_ts BETWEEN '{start}' AND '{end}'
GROUP BY entity_id, event_date, label
"""


def build_features(cfg: dict) -> pd.DataFrame:
    client  = bigquery.Client(project=cfg["project"])
    query   = FEATURE_QUERY.format(**cfg["data"])
    log.info("feature_query_started", project=cfg["project"])
    df = client.query(query).to_dataframe()
    log.info("features_built", rows=len(df), cols=len(df.columns))
    return df


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    df = build_features(cfg)
    out = cfg.get("output_path", "data/features.parquet")
    df.to_parquet(out, index=False)
    log.info("features_saved", path=out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    main(parser.parse_args().config)
