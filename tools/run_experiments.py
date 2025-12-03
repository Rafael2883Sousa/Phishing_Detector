# tools/run_experiments.py
#
# Runs experiments:
#   E0: text-only TF-IDF + LR
#   E1: text + URL features
#   E2: text + header features
#   E3: text + HTML feature (anchor_mismatch)
#
# Uses data/processed/emails_full.csv and writes:
#   outputs/E{n}_metrics.json
#   outputs/E{n}_test_predictions_email.csv
# Saves E0 pipeline + PR curve to outputs_email/.

import json
import re
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy import sparse
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from features.url_signals import url_features
    from features.headers import parse_headers
    from features.html_url import anchor_mismatch
except ImportError as e:
    raise ImportError(
        "Could not import features.url_signals/headers/html_url. "
        "Check that src/features/*.py exist and that src/__init__.py is present."
    ) from e


DATA_PATH = ROOT / "data" / "processed" / "emails_full.csv"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_EMAIL_DIR = ROOT / "outputs_email"
OUTPUTS_DIR.mkdir(exist_ok=True)
OUTPUTS_EMAIL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
RECALL_FLOOR = 0.90  # minimum recall for phishing class


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"id", "label", "subject", "body", "date", "headers_raw", "html"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    allowed = {"ham", "phishing"}
    bad = set(df["label"].unique()) - allowed
    if bad:
        raise ValueError(f"Unexpected labels found: {bad}")

    return df


def train_test_split_stratified(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple stratified split by label.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def build_text_fields(df: pd.DataFrame) -> np.ndarray:
    texts = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
    return texts.to_numpy()


URL_REGEX = re.compile(r"https?://\S+", flags=re.IGNORECASE)


def extract_urls_from_text(text: str) -> List[str]:
    return URL_REGEX.findall(text or "")


def aggregate_url_features(urls: List[str]) -> dict:
    """
    Aggregate URL features across all URLs in an email.
    Numeric features -> mean; boolean features -> any().
    If no URLs, all zeros/False.
    """
    if not urls:
        return {
            "len_url": 0.0,
            "len_host": 0.0,
            "len_path": 0.0,
            "num_dots": 0.0,
            "num_digits": 0.0,
            "num_subdomains": 0.0,
            "has_at": False,
            "has_double_slash": False,
            "tld_suspect": False,
            "confusable": False,
        }

    feats_list = [url_features(u) for u in urls]
    sample = feats_list[0]
    numeric_keys = [k for k, v in sample.items() if isinstance(v, (int, float))]
    bool_keys = [k for k, v in sample.items() if isinstance(v, bool)]

    agg = {}
    for k in numeric_keys:
        values = [float(f[k]) for f in feats_list]
        agg[k] = float(np.mean(values))

    for k in bool_keys:
        values = [bool(f[k]) for f in feats_list]
        agg[k] = bool(any(values))

    return agg


def build_url_feature_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    texts = build_text_fields(df)
    all_feats = []
    for txt in texts:
        urls = extract_urls_from_text(txt)
        feats = aggregate_url_features(urls)
        all_feats.append(feats)

    keys = [
        "len_url",
        "len_host",
        "len_path",
        "num_dots",
        "num_digits",
        "num_subdomains",
        "has_at",
        "has_double_slash",
        "tld_suspect",
        "confusable",
    ]
    mat = np.zeros((len(all_feats), len(keys)), dtype=float)
    for i, feats in enumerate(all_feats):
        for j, k in enumerate(keys):
            v = feats.get(k, 0.0)
            mat[i, j] = float(v) if not isinstance(v, bool) else float(v)

    return sparse.csr_matrix(mat)


def build_header_feature_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    """
    Use parse_headers(headers_raw) to get dict of boolean flags.
    """
    all_feats = []
    for raw in df["headers_raw"].fillna(""):
        feats = parse_headers(str(raw))
        all_feats.append(feats)

    # Determine consistent key order from first sample
    sample = all_feats[0]
    keys = list(sample.keys())
    mat = np.zeros((len(all_feats), len(keys)), dtype=float)
    for i, feats in enumerate(all_feats):
        for j, k in enumerate(keys):
            v = feats.get(k, False)
            mat[i, j] = float(v)

    return sparse.csr_matrix(mat)


def build_html_feature_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    """
    One simple boolean feature: anchor_mismatch(html).
    """
    vals = []
    for html in df["html"].fillna(""):
        mismatch = bool(anchor_mismatch(str(html)))
        vals.append(float(mismatch))
    mat = np.array(vals, dtype=float).reshape(-1, 1)
    return sparse.csr_matrix(mat)


def select_threshold(y_true: np.ndarray, scores: np.ndarray, recall_floor: float):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    candidates = []
    for j, thr in enumerate(thresholds):
        p = precision[j + 1]
        r = recall[j + 1]
        if r >= recall_floor:
            f1 = 2 * p * r / (p + r + 1e-15)
            candidates.append((thr, p, r, f1))

    if not candidates:
        best_f1 = -1.0
        best_tuple = None
        for j, thr in enumerate(thresholds):
            p = precision[j + 1]
            r = recall[j + 1]
            f1 = 2 * p * r / (p + r + 1e-15)
            if f1 > best_f1:
                best_f1 = f1
                best_tuple = (thr, p, r, f1)
        return best_tuple

    candidates.sort(key=lambda x: (x[1], x[3]), reverse=True)
    return candidates[0]


def evaluate_and_save(
    exp_name: str,
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_test: sparse.csr_matrix,
    y_test: np.ndarray,
    test_ids: np.ndarray,
    out_prefix: Path,
    save_pr_curve_path: Path | None = None,
):
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)[:, 1]
    else:
        scores = clf.decision_function(X_test)

    thr, p_thr, r_thr, f1_thr = select_threshold(
        y_test, scores, recall_floor=RECALL_FLOOR
    )

    y_pred = (scores >= thr).astype(int)

    ap = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    report = classification_report(
        y_test,
        y_pred,
        target_names=["ham", "phishing"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "experiment": exp_name,
        "best": {
            "threshold": float(thr),
            "precision_at_thr": float(p_thr),
            "recall_at_thr": float(r_thr),
            "f1_plus_at_thr": float(f1_thr),
            "average_precision": float(ap),
            "roc_auc": float(roc_auc),
            "test_size": 0.2,
            "n_test": int(len(y_test)),
            "n_train": int(len(y_train)),
            "class_distribution_test": {
                "ham": int((y_test == 0).sum()),
                "phishing": int((y_test == 1).sum()),
            },
            "class_distribution_train": {
                "ham": int((y_train == 0).sum()),
                "phishing": int((y_train == 1).sum()),
            },
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "recall_floor": float(RECALL_FLOOR),
        },
        "report": report,
    }

    metrics_path = out_prefix.with_name(out_prefix.name + "_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds_df = pd.DataFrame(
        {
            "id": test_ids,
            "y_true": y_test,
            "score": scores,
        }
    )
    preds_path = out_prefix.with_name(out_prefix.name + "_test_predictions_email.csv")
    preds_df.to_csv(preds_path, index=False)

    if save_pr_curve_path is not None:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        plt.figure()
        plt.step(recall, precision, where="post")
        plt.xlabel("Recall (phishing)")
        plt.ylabel("Precision (phishing)")
        plt.title(f"Precision-Recall Curve - {exp_name}")
        plt.grid(True)
        save_pr_curve_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_pr_curve_path, bbox_inches="tight")
        plt.close()

    print(f"[{exp_name}] metrics saved to: {metrics_path}")
    print(f"[{exp_name}] predictions saved to: {preds_path}")
    return clf

def evaluate_and_save_with_clf(
    exp_name: str,
    clf,
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_test: sparse.csr_matrix,
    y_test: np.ndarray,
    test_ids: np.ndarray,
    out_prefix: Path,
    save_pr_curve_path: Path | None = None,
):
    """
    Igual a evaluate_and_save, mas recebe o classificador (clf) de fora.
    """
    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)[:, 1]
    else:
        scores = clf.decision_function(X_test)

    thr, p_thr, r_thr, f1_thr = select_threshold(
        y_test, scores, recall_floor=RECALL_FLOOR
    )

    y_pred = (scores >= thr).astype(int)

    ap = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    report = classification_report(
        y_test,
        y_pred,
        target_names=["ham", "phishing"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "experiment": exp_name,
        "best": {
            "threshold": float(thr),
            "precision_at_thr": float(p_thr),
            "recall_at_thr": float(r_thr),
            "f1_plus_at_thr": float(f1_thr),
            "average_precision": float(ap),
            "roc_auc": float(roc_auc),
            "test_size": 0.2,
            "n_test": int(len(y_test)),
            "n_train": int(len(y_train)),
            "class_distribution_test": {
                "ham": int((y_test == 0).sum()),
                "phishing": int((y_test == 1).sum()),
            },
            "class_distribution_train": {
                "ham": int((y_train == 0).sum()),
                "phishing": int((y_train == 1).sum()),
            },
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "recall_floor": float(RECALL_FLOOR),
        },
        "report": report,
    }

    metrics_path = out_prefix.with_name(out_prefix.name + "_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds_df = pd.DataFrame(
        {
            "id": test_ids,
            "y_true": y_test,
            "score": scores,
        }
    )
    preds_path = out_prefix.with_name(out_prefix.name + "_test_predictions_email.csv")
    preds_df.to_csv(preds_path, index=False)

    if save_pr_curve_path is not None:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        plt.figure()
        plt.step(recall, precision, where="post")
        plt.xlabel("Recall (phishing)")
        plt.ylabel("Precision (phishing)")
        plt.title(f"Precision-Recall Curve - {exp_name}")
        plt.grid(True)
        save_pr_curve_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_pr_curve_path, bbox_inches="tight")
        plt.close()

    print(f"[{exp_name}] metrics saved to: {metrics_path}")
    print(f"[{exp_name}] predictions saved to: {preds_path}")
    return clf


# --- Embeddings (E5) ---

_EMB_MODEL: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        # MiniLM standard Model 
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


def build_embedding_matrix(texts: np.ndarray) -> sparse.csr_matrix:
    """
    Compute MiniLM sentence embeddings and return as csr_matrix.
    """
    model = get_embedding_model()
    # embeddings: shape (n_samples, dim)
    emb = model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return sparse.csr_matrix(emb)


def main():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)

    print("Splitting train/test...")
    train_df, test_df = train_test_split_stratified(df, test_size=0.2)

    label_to_int = {"ham": 0, "phishing": 1}
    y_train = train_df["label"].map(label_to_int).to_numpy()
    y_test = test_df["label"].map(label_to_int).to_numpy()
    test_ids = test_df["id"].to_numpy()

    print("Building text fields...")
    train_texts = build_text_fields(train_df)
    test_texts = build_text_fields(test_df)

    # Shared TF-IDF vectorizer (for E0–E3) for comparability
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X_train_text = vectorizer.fit_transform(train_texts)
    X_test_text = vectorizer.transform(test_texts)

    # E0: text-only
    print("Running E0 (text-only TF-IDF + LR)...")
    e0_prefix = OUTPUTS_DIR / "E0"
    e0_pr_curve_path = OUTPUTS_EMAIL_DIR / "pr_curve.png"
    clf_e0 = evaluate_and_save(
        exp_name="E0_text_only",
        X_train=X_train_text,
        y_train=y_train,
        X_test=X_test_text,
        y_test=y_test,
        test_ids=test_ids,
        out_prefix=e0_prefix,
        save_pr_curve_path=e0_pr_curve_path,
    )

    # Save E0 pipeline for API usage
    print("Saving E0 pipeline to outputs_email/tfidf_logreg_pipeline.joblib ...")
    e0_pipe = Pipeline([("tfidf", vectorizer), ("clf", clf_e0)])
    dump(e0_pipe, OUTPUTS_EMAIL_DIR / "tfidf_logreg_pipeline.joblib")
    print("E0 pipeline saved.")

    # E1: text + URL
    print("Running E1 (text + URL features)...")
    X_train_url = build_url_feature_matrix(train_df)
    X_test_url = build_url_feature_matrix(test_df)
    X_train_e1 = sparse.hstack([X_train_text, X_train_url], format="csr")
    X_test_e1 = sparse.hstack([X_test_text, X_test_url], format="csr")
    e1_prefix = OUTPUTS_DIR / "E1"
    _ = evaluate_and_save(
        exp_name="E1_text_plus_url",
        X_train=X_train_e1,
        y_train=y_train,
        X_test=X_test_e1,
        y_test=y_test,
        test_ids=test_ids,
        out_prefix=e1_prefix,
        save_pr_curve_path=None,
    )

    # E2: text + headers
    print("Running E2 (text + headers)...")
    X_train_hdr = build_header_feature_matrix(train_df)
    X_test_hdr = build_header_feature_matrix(test_df)
    X_train_e2 = sparse.hstack([X_train_text, X_train_hdr], format="csr")
    X_test_e2 = sparse.hstack([X_test_text, X_test_hdr], format="csr")
    e2_prefix = OUTPUTS_DIR / "E2"
    _ = evaluate_and_save(
        exp_name="E2_text_plus_headers",
        X_train=X_train_e2,
        y_train=y_train,
        X_test=X_test_e2,
        y_test=y_test,
        test_ids=test_ids,
        out_prefix=e2_prefix,
        save_pr_curve_path=None,
    )

    # E3: text + HTML (anchor_mismatch)
    print("Running E3 (text + HTML feature)...")
    X_train_html = build_html_feature_matrix(train_df)
    X_test_html = build_html_feature_matrix(test_df)
    X_train_e3 = sparse.hstack([X_train_text, X_train_html], format="csr")
    X_test_e3 = sparse.hstack([X_test_text, X_test_html], format="csr")
    e3_prefix = OUTPUTS_DIR / "E3"
    _ = evaluate_and_save(
        exp_name="E3_text_plus_html",
        X_train=X_train_e3,
        y_train=y_train,
        X_test=X_test_e3,
        y_test=y_test,
        test_ids=test_ids,
        out_prefix=e3_prefix,
        save_pr_curve_path=None,
    )

    # E4: TF-IDF + LinearSVC (SVM linear)
    print("Running E4 (text-only TF-IDF + LinearSVC)...")
    clf_e4 = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    e4_prefix = OUTPUTS_DIR / "E4"
    _ = evaluate_and_save_with_clf(
        exp_name="E4_text_only_LinearSVC",
        clf=clf_e4,
        X_train=X_train_text,
        y_train=y_train,
        X_test=X_test_text,
        y_test=y_test,
        test_ids=test_ids,
        out_prefix=e4_prefix,
        save_pr_curve_path=None,
    )

    # E5: MiniLM embeddings + Logistic Regression
    print("Running E5 (MiniLM embeddings + LR)...")
    # Reusar os mesmos textos (subject+body)
    X_train_emb = build_embedding_matrix(train_texts)
    X_test_emb = build_embedding_matrix(test_texts)

    clf_e5 = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    e5_prefix = OUTPUTS_DIR / "E5"
    _ = evaluate_and_save_with_clf(
        exp_name="E5_embeddings_MiniLM_LR",
        clf=clf_e5,
        X_train=X_train_emb,
        y_train=y_train,
        X_test=X_test_emb,
        y_test=y_test,
        test_ids=test_ids,
        out_prefix=e5_prefix,
        save_pr_curve_path=None,
    )

    print("All experiments E0–E5 finished.")

if __name__ == "__main__":
    main()
