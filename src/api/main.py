# src/api/main.py
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List, Optional
from src.rules.engine import load_rule_engine

from pydantic import BaseModel
from joblib import load
from fastapi import FastAPI

from src.features.url_signals import url_features
from src.features.html_url import anchor_mismatch
from src.features.headers import parse_headers

import logging

SRC_DIR = Path(__file__).resolve().parents[1]   
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("phishing_api")


PIPE_PATH   = os.environ.get("PIPE_PATH", "outputs_email/tfidf_logreg_pipeline.joblib")
# pipe_sms = load(os.environ.get("PIPE_SMS", "outputs_sms/tfidf_logreg_pipeline.joblib"))

REPORT_PATH = os.environ.get("REPORT_PATH", "outputs_email/baseline_report.json")

try:
    pipe = load(PIPE_PATH)
except Exception as e:
    logger.warning("ML pipeline not loaded: %s", e)
    pipe = None

logger.info("Loaded ML pipeline with classes: %s", list(pipe.classes_))

THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
try:
    with open(REPORT_PATH, "r") as f:
        THRESHOLD = float(json.load(f)["best"]["threshold"])
except Exception:
    pass

app = FastAPI()
rule_engine = load_rule_engine()

class EmailInput(BaseModel):
    subject: Optional[str] = ""
    body: Optional[str] = ""
    urls: Optional[List[str]] = []
    headers_raw: Optional[str] = ""
    html: Optional[str] = ""

class SmsInput(BaseModel):
    text: str
    urls: list[str] | None = None

def _join(i: EmailInput) -> str:
    return f"{i.subject or ''} {i.body or ''}".strip()

def _reasons(i: EmailInput) -> List[str]:
    r: List[str] = []
    for u in (i.urls or []):
        feats = url_features(u)
        r += [k for k, v in feats.items() if isinstance(v, bool) and v]
    if i.html and anchor_mismatch(i.html):
        r.append("anchor_href_mismatch")
    if i.headers_raw:
        h = parse_headers(i.headers_raw)
        r += [k for k, v in h.items() if v]
    return sorted(set(r))

@app.get("/")
def root():
    return {"status": "ok", "threshold": THRESHOLD, "endpoints": ["/predict", "/docs"]}

@app.post("/predict")
def predict(i: EmailInput):
    text = _join(i)
    reasons = _reasons(i)

    proba = None
    label = "ham"
    decision_source = "rules"

    if pipe is not None:
        classes = list(pipe.classes_)

        if "phishing" in classes:
            pos_label = "phishing"
        elif "spam" in classes:
            pos_label = "spam"
        elif 1 in classes:
            pos_label = 1
        else:
            raise RuntimeError(f"Cannot determine positive class from {classes}")

        ph_idx = classes.index(pos_label)

        proba = float(pipe.predict_proba([text])[0, ph_idx])
        label = "phishing" if proba >= THRESHOLD else "ham"
        decision_source = "ml+rules"

    rule_out = rule_engine.run({
        "subject": i.subject,
        "body": i.body,
        "urls": i.urls,
        "headers_raw": i.headers_raw,
        "html": i.html,
    })

    reasons = sorted(set(reasons + rule_out.get("reasons", [])))

    if proba is None:
        proba = rule_out.get("risk_score", 0.0)
        label = "phishing" if proba >= 0.5 else "ham"

    logger.info(
        "EMAIL PREDICT score=%.4f label=%s source=%s reasons=%s",
        proba,
        label,
        decision_source,
        ",".join(reasons),
    )

    return {
        "score": proba,
        "label": label,
        "reasons": reasons,
        "decision_source": decision_source,
    }

# @app.post("/predict_sms")
# def predict_sms(i: SmsInput):
#     proba = float(pipe_sms.predict_proba([i.text])[0, list(pipe_sms.classes_).index("phishing")])
#     label = "phishing" if proba >= THRESHOLD else "ham"
#     return {"score": proba, "label": label, "reasons": []}