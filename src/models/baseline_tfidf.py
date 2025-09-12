"""
Baseline text classifier with TF-IDF + Logistic Regression and PR-curve.

Assumptions:
- Python 3.11, CPU.
- Input CSV follows one of the schemas:
  * email: id,label,subject,body[,urls,from,reply_to,return_path,auth_results]
  * sms:   id,label,text[,urls,sender]
- Labels accepted: "ham" or "phishing" (or 0/1 which are remapped).

Output artifacts (in --outdir):
- pr_curve.png
- baseline_report.json
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

from joblib import dump
from sklearn.pipeline import Pipeline


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV and return (texts, labels)."""
    df = pd.read_csv(path)
    # normalize label to string
    y = df["label"].astype(str).str.lower().replace(
        {"spam": "phishing", "0": "ham", "1": "phishing"}
    )

    # pick text column(s)
    if "text" in df.columns:
        texts = df["text"].astype(str)
    else:
        subject = df.get("subject", "").astype(str)
        body = df.get("body", "").astype(str)
        texts = (subject + " " + body).astype(str)

    return texts.values, y.values


def choose_test_size(labels: np.ndarray) -> float:
    """Choose a test_size that guarantees at least one sample per class in test."""
    # minimum robust split when dataset is tiny
    vc = pd.Series(labels).value_counts()
    if (vc < 2).any():
        # too few samples per class -> use 50% to try to keep both classes in test
        return 0.5
    # default
    return 0.2


def train_eval(texts: np.ndarray, labels: np.ndarray, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    test_size = choose_test_size(labels)
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Vectorizer: allow small datasets to work (min_df=1)
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    Xtr = vect.fit_transform(Xtr_txt)
    Xte = vect.transform(Xte_txt)

    # Small/CPU-friendly solver
    clf = LogisticRegression(
        max_iter=200, class_weight="balanced", solver="liblinear"
    )
    clf.fit(Xtr, ytr)

    # Save artefacts
    dump(vect, os.path.join(outdir, "tfidf_vectorizer.joblib"))
    dump(clf,  os.path.join(outdir, "logreg_model.joblib"))

    # Unique Pipeline
    pipe = Pipeline([("tfidf", vect), ("clf", clf)])
    dump(pipe, os.path.join(outdir, "tfidf_logreg_pipeline.joblib"))

    # Ensure "phishing" exists in classes
    classes = list(clf.classes_)
    if "phishing" not in classes:
        raise ValueError(
            "Training split is missing the 'phishing' class. "
            "Increase dataset size or adjust the split."
        )
    ph_idx = int(np.where(np.array(classes) == "phishing")[0][0])

    # Ground truth and scores
    y_true = (yte == "phishing").astype(int)
    y_score = clf.predict_proba(Xte)[:, ph_idx]

    # PR curve and AP
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # F1 per PR point; thresholds align to indices 1..N
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    # If all-zero or nan, guard with fallback
    if np.all(np.isnan(f1)):
        best_thr = float(np.quantile(y_score, 0.5))
    else:
        # best index among points that have a corresponding threshold
        if len(thresholds) > 0:
            best_i = int(np.nanargmax(f1[1:])) + 1  # map to precision/recall index
            best_thr = float(thresholds[best_i - 1])
        else:
            best_thr = 0.5

    # Predictions at chosen threshold
    y_pred = (y_score >= best_thr).astype(int)

    # Fallback: if no positives predicted, relax threshold
    if y_pred.sum() == 0:
        best_thr = float(np.quantile(y_score, 0.8))
        y_pred = (y_score >= best_thr).astype(int)

    # Report with zero_division=0 to avoid warnings on degenerate small sets
    report = classification_report(
        y_true,
        y_pred,
        target_names=["ham", "phishing"],
        output_dict=True,
        zero_division=0,
    )

    # Best summary at threshold
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    pp = int((y_pred == 1).sum())
    p1 = int((y_true == 1).sum())
    prec_at_thr = float(tp / pp) if pp > 0 else 0.0
    rec_at_thr = float(tp / p1) if p1 > 0 else 0.0
    f1_at_thr = (
        (2 * prec_at_thr * rec_at_thr) / (prec_at_thr + rec_at_thr + 1e-12)
        if (prec_at_thr + rec_at_thr) > 0
        else 0.0
    )

    best = {
        "threshold": best_thr,
        "precision_at_thr": prec_at_thr,
        "recall_at_thr": rec_at_thr,
        "f1_plus_at_thr": f1_at_thr,
        "average_precision": float(ap),
        "test_size": test_size,
        "n_test": int(len(yte)),
        "n_train": int(len(ytr)),
        "class_distribution_test": {
            "ham": int((y_true == 0).sum()),
            "phishing": int((y_true == 1).sum()),
        },
    }

    # Save artifacts
    with open(os.path.join(outdir, "baseline_report.json"), "w") as f:
        json.dump({"best": best, "report": report}, f, indent=2)

    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall — TF-IDF + Logistic Regression")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with (subject/body) or (text) and label.",
    )
    ap.add_argument(
        "--outdir", default="outputs", help="Directory for reports and plots."
    )
    args = ap.parse_args()
    texts, labels = load_data(args.csv)
    train_eval(texts, labels, args.outdir)


if __name__ == "__main__":
    main()
