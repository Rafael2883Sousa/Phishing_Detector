# --- top ---
from joblib import load
import os, json
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# features
from src.features.url_signals import url_features
from src.features.html_url import anchor_mismatch
from src.features.headers import parse_headers

PIPE_PATH = os.environ.get("PIPE_PATH", "outputs_email/tfidf_logreg_pipeline.joblib")
REPORT_PATH = os.environ.get("REPORT_PATH", "outputs_email/baseline_report.json")

pipe = load(PIPE_PATH)

# threshold: env > report.json > 0.5
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

def join_text_email(i: EmailInput) -> str:
    return f"{i.subject or ''} {i.body or ''}".strip()

def collect_reasons(i: EmailInput) -> List[str]:
    reasons = []
    # URL features
    for u in (i.urls or []):
        feats = url_features(u)
        reasons += [k for k, v in feats.items() if isinstance(v, bool) and v]
    # HTML anchor vs href
    if i.html and anchor_mismatch(i.html):
        reasons.append("anchor_href_mismatch")
    # Headers flags
    if i.headers_raw:
        h = parse_headers(i.headers_raw)
        reasons += [k for k, v in h.items() if v]
    return list(sorted(set(reasons)))

@app.post("/predict")
def predict(i: EmailInput):
    text = join_text_email(i)
    proba = float(pipe.predict_proba([text])[0, list(pipe.classes_).index("phishing")])
    label = "phishing" if proba >= THRESHOLD else "ham"
    reasons = collect_reasons(i)
    return {"score": proba, "label": label, "reasons": reasons}

@app.get("/")
def root():
    return {"status": "ok", "threshold": THRESHOLD, "endpoints": ["/predict", "/docs"]}
