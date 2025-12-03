# -*- coding: utf-8 -*-
# Compute F1+ and Recall+ per robustness scenario using fixed threshold t.
import argparse, json, pandas as pd, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

SCENARIOS = [
    ("Homógrafos Unicode", "unicode_homoglyph == True"),
    ("URLs encurtadas",    "shortened_url == True"),
    ("Redireções em cadeia","redirector_url == True"),
    ("PT-BR vs PT-PT",     "ptbr_likely == True"),
    ("Mensagens sem URL",  "no_url == True"),
    ("Headers ausentes",   "headers_missing == True"),
]

def subset_metrics(df, t):
    y = df["y_true"].values.astype(int)
    y_pred = (df["score"].values >= t).astype(int)
    return {
        "N": int(len(df)),
        "Precision+": float(precision_score(y, y_pred, zero_division=0)),
        "Recall+": float(recall_score(y, y_pred, zero_division=0)),
        "F1+": float(f1_score(y, y_pred, zero_division=0)),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)  # outputs/E3_test_predictions.csv
    ap.add_argument("--flags_csv", required=True) # outputs/test_flags.csv
    ap.add_argument("--threshold_json", required=True)  # outputs/E3_metrics.json
    ap.add_argument("--out_csv", required=True)  # outputs/robustness_E3.csv
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)          # id,y_true,score
    flags = pd.read_csv(args.flags_csv)        # id, flags...
    meta = json.load(open(args.threshold_json,"r"))
    t = float(meta["Limiar"])

    df = pred.merge(flags, on="id", how="inner")

    rows = []
    for name, query in SCENARIOS:
        sub = df.query(query)
        if len(sub)==0:
            rows.append({"Cenário": name, "Descrição": query, "Métrica": "F1+", "Valor": "NA", "Notas": "N=0"})
            continue
        m = subset_metrics(sub, t)
        rows.append({"Cenário": name, "Descrição": query, "Métrica": "F1+", "Valor": round(m["F1+"],4),
                     "Notas": f'N={m["N"]}; Recall+={m["Recall+"]:.4f}; Precision+={m["Precision+"]:.4f}; t={t:.4f}'})
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"gravado: {args.out_csv}")
