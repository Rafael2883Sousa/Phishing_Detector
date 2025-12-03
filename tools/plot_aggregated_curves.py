import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc


# Base directory for CSV prediction files (adjust if needed)
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs"

# Experiments to aggregate: name -> CSV file
EXPERIMENTS = {
    "E0 – Texto": "E0_test_predictions_email.csv",
    "E1 – Texto+URL": "E1_test_predictions_email.csv",
    "E2 – Texto+Headers": "E2_test_predictions_email.csv",
    "E3 – Texto+HTML": "E3_test_predictions_email.csv",
    "E4 – SVM": "E4_test_predictions_email.csv",
    "E5 – Embeddings": "E5_test_predictions_email.csv",
}


def load_predictions(path: Path):
    """Load y_true and scores from CSV file."""
    df = pd.read_csv(path)
    if not {"y_true", "score"}.issubset(df.columns):
        raise ValueError(f"CSV {path} must have 'y_true' and 'score' columns.")
    y_true = df["y_true"].to_numpy()
    scores = df["score"].to_numpy()
    return y_true, scores


def plot_aggregated_pr_curves():
    """Plot all Precision–Recall curves on the same figure."""
    plt.figure()
    for label, filename in EXPERIMENTS.items():
        csv_path = BASE_DIR / filename
        if not csv_path.exists():
            print(f"[WARN] File not found, skipping PR: {csv_path}")
            continue

        y_true, scores = load_predictions(csv_path)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)

        # Sort recall/precision by recall for consistent plotting
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]

        plt.step(recall_sorted, precision_sorted, where="post", label=f"{label} (AUC={pr_auc:.3f})")

    plt.xlabel("Recall (classe phishing)")
    plt.ylabel("Precisão (classe phishing)")
    plt.title("Curvas Precision–Recall agregadas (E0–E5)")
    plt.grid(True)
    plt.legend(loc="lower left", fontsize="small")

    out_path = BASE_DIR / "PR_curves_E0_E5.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Figura PR gravada em: {out_path}")


def plot_aggregated_roc_curves():
    """Plot all ROC curves on the same figure."""
    plt.figure()

    # Diagonal baseline
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Baseline aleatório")

    for label, filename in EXPERIMENTS.items():
        csv_path = BASE_DIR / filename
        if not csv_path.exists():
            print(f"[WARN] File not found, skipping ROC: {csv_path}")
            continue

        y_true, scores = load_predictions(csv_path)
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC agregadas (E0–E5)")
    plt.grid(True)
    plt.legend(loc="lower right", fontsize="small")

    out_path = BASE_DIR / "ROC_curves_E0_E5.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Figura ROC gravada em: {out_path}")


def main():
    print(f"[INFO] Base de ficheiros: {BASE_DIR}")
    plot_aggregated_pr_curves()
    plot_aggregated_roc_curves()


if __name__ == "__main__":
    main()
