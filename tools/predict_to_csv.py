# -*- coding: utf-8 -*-
# Usage:
#   python tools/predict_to_csv.py --pipe outputs_email/tfidf_logreg_pipeline.joblib \
#                                  --csv_test data/processed/mini_sample.csv \
#                                  --out outputs/E0_test_predictions.csv \
#                                  --mode email
import argparse, pandas as pd, numpy as np, joblib, os

def load_text(row, mode):
    if mode == "email":
        subj = str(row.get("subject","") or "")
        body = str(row.get("body","") or "")
        return (subj + " " + body).strip()
    else:  # sms
        return str(row.get("text","") or "").strip()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", required=True, help="joblib pipeline path")
    ap.add_argument("--csv_test", required=True, help="CSV com dados rotulados")
    ap.add_argument("--out", required=True, help="CSV de saída com id,y_true,score")
    ap.add_argument("--mode", choices=["email","sms"], default="email")
    ap.add_argument("--label_col", default="label", help="coluna rótulo: phishing/legit ou 1/0")
    args = ap.parse_args()

    pipe = joblib.load(args.pipe)
    df = pd.read_csv(args.csv_test)
    if args.label_col not in df.columns:
        raise SystemExit(f"Coluna de rótulo '{args.label_col}' não existe em {args.csv_test}")

    # normaliza rótulo para {0,1} onde 1=phishing
    ymap = {"phishing":1, "legit":0, "ham":0, "spam":1}
    y_true = df[args.label_col].apply(lambda v: int(ymap.get(str(v).strip().lower(), v))).astype(int)

    Xtxt = df.apply(lambda r: load_text(r, args.mode), axis=1).tolist()
    # obtém score: usar predict_proba se existir, senão decision_function
    if hasattr(pipe, "predict_proba"):
        score = pipe.predict_proba(Xtxt)[:,1]
    elif hasattr(pipe, "decision_function"):
        # mapeia decisão para [0,1] via logística
        s = pipe.decision_function(Xtxt)
        score = 1/(1+np.exp(-s))
    else:
        # fallback: usa predict como 0/1 já que pior das hipóteses
        score = pipe.predict(Xtxt).astype(float)

    out = pd.DataFrame({
        "id": np.arange(len(df)),
        "y_true": y_true.values,
        "score": score
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"gravado: {args.out} ({len(out)} linhas)")
