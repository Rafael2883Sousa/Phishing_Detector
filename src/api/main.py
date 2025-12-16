# src/api/main.py
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List, Optional, Literal
from src.rules.engine import load_rule_engine
from src.rules.engine import extract_urls

from pydantic import BaseModel, Field
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
    id: Optional[str] = Field(
        None,
        description="Identificador opcional da mensagem"
    )
    subject: str = Field(
        ...,
        description="Assunto do email"
    )
    body: str = Field(
        ...,
        description="Corpo textual do email"
    )
    headers_raw: Optional[str] = Field(
        None,
        description="Cabeçalhos completos do email (Authentication-Results, From, etc.)"
    )
    html: Optional[str] = Field(
        None,
        description="Conteúdo HTML bruto do email"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "subject": "Atualização de segurança da conta",
                "body": "Por favor confirme os seus dados no portal.",
                "headers_raw": "From: suporte@empresa.pt",
                "html": "<a href='http://exemplo.com'>empresa.pt</a>"
            }
        }
    }


class FeedbackInput(BaseModel):
    id: str
    label_true: str 
    context: Optional[str] = None 

class PredictResponse(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada da classe phishing"
    )
    label: Literal["phishing", "legit"] = Field(
        ...,
        description="Decisão final após aplicação do limiar"
    )
    reasons: List[str] = Field(
        ...,
        description="Sinais estruturais que contribuíram para a decisão"
    )

class PredictionOutput(BaseModel):
    score: float
    label: str
    reasons: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "score": 0.73,
                "label": "phishing",
                "reasons": [
                    "tld_suspect",
                    "anchor_href_mismatch",
                    "from_reply_mismatch"
                ]
            }
        }
    }

# class SmsInput(BaseModel):
#     text: str
#     urls: list[str] | None = None

def _join(i: EmailInput) -> str:
    return f"{i.subject or ''} {i.body or ''}".strip()

def _reasons(i: EmailInput) -> list[str]:
    reasons: list[str] = []

    urls = _extract_urls(i)

    for u in urls:
        feats = url_features(u)
        reasons += [k for k, v in feats.items() if isinstance(v, bool) and v]

    if i.html and anchor_mismatch(i.html):
        reasons.append("anchor_href_mismatch")

    if i.headers_raw:
        h = parse_headers(i.headers_raw)
        reasons += [k for k, v in h.items() if v]

    return sorted(set(reasons))


def _extract_urls(i: EmailInput) -> list[str]:
    text = f"{i.subject} {i.body}"
    urls = extract_urls(text)
    if i.html:
        urls += extract_urls(i.html)
    return list(set(urls))

@app.get("/")
def root():
    return {"status": "ok", "threshold": THRESHOLD, "endpoints": ["/predict", "/docs"]}

@app.post("/predict",
    response_model=PredictionOutput,
    summary="Phishing predition (email)",
    description="""
    Avalia uma mensagem de email para deteção de phishing.

- Extração automática de URLs
- Modelo TF-IDF + classificador linear
- Limiar calibrado via curva Precision-Recall
- Explicações baseadas em regras determinísticas
""",
)
def predict(i: EmailInput):

    text = _join(i)

    classes = list(pipe.classes_)

    if "phishing" in classes:
        ph_idx = classes.index("phishing")

    elif 1 in classes:
        ph_idx = classes.index(1)

    else:
        raise RuntimeError(f"Pipeline classes not suported: {classes}")

    proba = float(pipe.predict_proba([text])[0, ph_idx])

    ml_phish = proba >= THRESHOLD

    rule_engine = load_rule_engine()

    rule_out = rule_engine.run({
        "subject": i.subject,
        "body": i.body,
        "headers_raw": i.headers_raw,
        "html": i.html,
    })

    reasons = rule_out.get("reasons", [])
    risk_score = float(rule_out.get("risk_score", 0.0))

    # --- Decisão final (ML + regras) ---

    # Caso 1: múltiplas regras acusatórias → override
    if risk_score >= 0.66:
        label = "phishing"
        decision_source = "rules_override"

    # Caso 2: ML confiante
    elif ml_phish:
        label = "phishing"
        decision_source = "ml"

    # Caso 3: borderline textual + pelo menos 1 regra
    elif proba >= 0.6 and risk_score > 0.0:
        label = "phishing"
        decision_source = "ml_borderline+rules"

    # Caso 4: benigno
    else:
        label = "legit"
        decision_source = "none"

    logger.info(
        "EMAIL PREDICT score=%.4f ml=%s rules=%s risk=%.2f label=%s reasons=%s",
        proba,
        ml_phish,
        risk_score,
        label,
        decision_source,
        ",".join(reasons),
    )

    return {
        "score": proba,
        "label": label,
        "reasons": reasons,
    }

FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "outputs/feedback.jsonl")

@app.post("/feedback")
def feedback(f: FeedbackInput):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

    record = {
        "id": f.id,
        "label_true": f.label_true,
        "context": f.context,
    }

    with open(FEEDBACK_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("FEEDBACK id=%s label=%s context=%s", f.id, f.label_true, f.context)

    return {"status": "ok"}

