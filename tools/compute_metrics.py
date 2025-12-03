import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score

def select_threshold(y_true, scores, recall_floor=0.90):
    p, r, th = precision_recall_curve(y_true, scores)
    # exclude last point where threshold is nan
    p, r, th = p[:-1], r[:-1], th
    # candidates with Recall >= floor
    mask = r >= recall_floor
    if not mask.any():
        # fallback: pick threshold at max F1
        f1 = (2*p*r)/(p+r+1e-12)
        t = th[f1.argmax()]
        return float(t), {"mode":"maxF1_fallback"}
    # among candidates, pick max Precision then max F1
    cand_idx = np.where(mask)[0]
    best = max(cand_idx, key=lambda i: (p[i], (2*p[i]*r[i])/(p[i]+r[i]+1e-12)))
    return float(th[best]), {"mode":"recall_floor", "precision":float(p[best]), "recall":float(r[best])}

def compute_all(y_true, scores, t):
    y_pred = (scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, scores)  # PR-AUC (AP)
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")
    return {
        "Limiar": t,
        "Precision+": prec,
        "Recall+": rec,
        "F1+": f1,
        "PR-AUC": pr_auc,
        "AUROC": auroc,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Suporte+": int(tp+fn), "Suporte-": int(tn+fp)
    }

def run(csv_path, recall_floor):
    df = pd.read_csv(csv_path)
    t, info = select_threshold(df["y_true"].values, df["score"].values, recall_floor)
    out = compute_all(df["y_true"].values, df["score"].values, t)
    out["Notas"] = json.dumps(info)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--recall_floor", type=float, default=0.90)
    args = ap.parse_args()
    res = run(args.csv, args.recall_floor)
    print(json.dumps(res, ensure_ascii=False))
