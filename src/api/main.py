# src/api/main.py
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from joblib import load

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

from features.url_signals import url_features
from features.html_url import anchor_mismatch
from features.headers import parse_headers

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

pipe = load(PIPE_PATH)

THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
try:
    with open(REPORT_PATH, "r") as f:
        THRESHOLD = float(json.load(f)["best"]["threshold"])
except Exception:
    pass

app = FastAPI()

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
    ph_idx = list(pipe.classes_).index("phishing")
    proba = float(pipe.predict_proba([text])[0, ph_idx])
    label = "phishing" if proba >= THRESHOLD else "ham"
    logger.info(
    "EMAIL PREDICT id=%s score=%.4f label=%s reasons=%s",
    getattr(email, "id", "NA"),
    float(score),
    label,
    ",".join(reasons),
)

    return {"score": proba, "label": label, "reasons": _reasons(i)}

# @app.post("/predict_sms")
# def predict_sms(i: SmsInput):
#     proba = float(pipe_sms.predict_proba([i.text])[0, list(pipe_sms.classes_).index("phishing")])
#     label = "phishing" if proba >= THRESHOLD else "ham"
#     return {"score": proba, "label": label, "reasons": []}