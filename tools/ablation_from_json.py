# -*- coding: utf-8 -*-
# Le E0..E5_metrics.json e produz deltas vs E0 para a tabela de Ablação.
import json, argparse, os, math
from pathlib import Path

def loadm(p): 
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[
        "outputs/E0_metrics.json","outputs/E1_metrics.json",
        "outputs/E2_metrics.json","outputs/E3_metrics.json",
        "outputs/E4_metrics.json","outputs/E5_metrics.json"])
    ap.add_argument("--out", default="outputs/ablation_E1_E5_vs_E0.csv")
    args = ap.parse_args()

    M = {}
    for p in args.inputs:
        k = Path(p).stem.split("_")[0].upper()  # E0, E1...
        M[k] = loadm(p)

    base = M["E0"]
    def delta(k,e): 
        a = float(M[e].get(k,float("nan"))); b = float(base.get(k,float("nan")))
        return None if (math.isnan(a) or math.isnan(b)) else round(a-b,4)

    rows = []
    for e in ["E1","E2","E3","E4","E5"]:
        rows.append({
          "Comparação": f"{e} − E0",
          "ΔPrecision⁺": delta("Precision+", e),
          "ΔRecall⁺":    delta("Recall+", e),
          "ΔF1⁺":        delta("F1+", e),
          "ΔPR-AUC":     delta("PR-AUC", e),
          "Observações": f"Limiar {e}={M[e]['Limiar']:.4f}; E0={base['Limiar']:.4f}"
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"gravado: {args.out}")
