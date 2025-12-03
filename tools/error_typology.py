import argparse, json, re, pandas as pd, numpy as np
from urllib.parse import urlsplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

URL_RGX = re.compile(r"https?://[^\s)>\]}\"']+", re.I)

def mask_domains(text):
    def repl(m):
        u = m.group(0)
        try: net = urlsplit(u).netloc
        except: net = "dominio.tld"
        parts = net.split(".")
        masked = "***." + ".".join(parts[-2:]) if len(parts)>=2 else "***."+net
        return u.replace(net, masked)
    return URL_RGX.sub(repl, str(text or ""))

def short_text(subject, body, maxlen=220):
    t = ((subject or "").strip() + " " + (body or "").strip()).strip()
    t = re.sub(r"\s+", " ", t)
    t = mask_domains(t)
    return (t[:maxlen] + "…") if len(t) > maxlen else t

def hdr_auth_ok(headers_json):
    try:
        if not isinstance(headers_json, str) or not headers_json.strip().startswith("{"):
            return False
        h = json.loads(headers_json)
    except Exception:
        return False
    text = json.dumps(h).lower()
    return all(k in text for k in ["spf","dkim","dmarc"]) and ("pass" in text)

def classify_error(row):
    if row.y_true == 0 and row.y_pred == 1:
        if row.ptbr_likely: return "FP_ptbr"
        if (not row.headers_missing) and row.hdr_auth_ok: return "FP_header_ok"
        return "FP_newsletter"
    if row.y_true == 1 and row.y_pred == 0:
        if row.no_url: return "FN_sem_url"
        if row.redirector_url or row.shortened_url: return "FN_redirect_chain"
        return "FN_compromised_domain"
    return "OK"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--threshold_json", required=True)
    ap.add_argument("--raw_csv", required=True)
    ap.add_argument("--flags_csv", required=True)
    ap.add_argument("--out_summary", default="outputs/error_typology_summary.csv")
    ap.add_argument("--out_examples", default="outputs/error_typology_examples.csv")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)          # id,y_true,score
    flags = pd.read_csv(args.flags_csv)        # id + flags
    raw = pd.read_csv(args.raw_csv)            # id,subject,body,headers_json?
    t = float(json.load(open(args.threshold_json))["Limiar"])

    # Normaliza rótulos
    ymap = {"1":1,"0":0,"phishing":1,"spam":1,"legit":0,"ham":0}
    pred["y_true"] = pred["y_true"].astype(str).str.lower().map(ymap).astype(int)

    # Merge INNER para garantir flags+texto
    df = pred.merge(flags, on="id", how="inner").merge(raw, on="id", how="inner")
    if "headers_json" in df.columns: df["hdr_auth_ok"] = df["headers_json"].apply(hdr_auth_ok)
    else: df["hdr_auth_ok"] = False

    df["y_pred"] = (df["score"] >= t).astype(int)
    df["tipo"] = df.apply(classify_error, axis=1)

    # Resumo por tipo
    g = df.groupby("tipo")["id"].count().rename("N").reset_index()

    # Se não há FP/FN, cria near-miss para exemplos
    examples = []
    if not any(df["tipo"].isin(["FP_newsletter","FP_ptbr","FP_header_ok","FN_compromised_domain","FN_redirect_chain","FN_sem_url"])):
        near = df.loc[(df["score"]-t).abs()<=0.05].copy()
        near["tipo"] = np.where(near["y_true"]==1, "FN_quase","FP_quase")
        for _,r in near.head(6).iterrows():
            examples.append({
                "Tipo de erro": r["tipo"],
                "Exemplo anonimizado": short_text(r.get("subject",""), r.get("body","")),
                "Sinal ausente/insuficiente": (
                    "URL/redirect" if (r.get("redirector_url",False) or r.get("shortened_url",False) or r.get("no_url",False)) else
                    "Headers/auth" if (not r.get("headers_missing",True) and r.get("hdr_auth_ok",False)) else
                    "Lexical/contexto"
                ),
                "Mitigação sugerida": (
                    "Apertar análise de URL/QR/redirect; revisão humana de zona cinzenta" if r["tipo"].startswith("FN_") else
                    "Allowlist e ponderação de cabeçalhos; ajuste fino de limiar"
                )
            })
    else:
        # Exemplos reais por tipo
        for etype in ["FP_newsletter","FP_ptbr","FP_header_ok","FN_compromised_domain","FN_redirect_chain","FN_sem_url"]:
            sub = df[df["tipo"] == etype].head(3)
            for _,r in sub.iterrows():
                examples.append({
                    "Tipo de erro": etype,
                    "Exemplo anonimizado": short_text(r.get("subject",""), r.get("body","")),
                    "Sinal ausente/insuficiente": (
                        "URL/redirect" if etype in ["FN_redirect_chain","FN_sem_url"] else
                        "Headers/auth" if etype=="FP_header_ok" else
                        "Lexical PT-PT" if etype=="FP_ptbr" else
                        "Regras newsletter/allowlist" if etype=="FP_newsletter" else
                        "Domínio comprometido/reputação"
                    ),
                    "Mitigação sugerida": (
                        "Detetar QR/redirect e impor análise de destino" if etype in ["FN_redirect_chain","FN_sem_url"] else
                        "Usar SPF/DKIM/DMARC na decisão e coerência From/Reply-To" if etype=="FP_header_ok" else
                        "Normalização PT-BR; reforçar sinais de URL" if etype=="FP_ptbr" else
                        "Allowlist e ajuste de limiar" if etype=="FP_newsletter" else
                        "Feed de domínios comprometidos; âncora≠href"
                    )
                })

    # Guarda
    g.to_csv(args.out_summary, index=False)
    pd.DataFrame(examples).to_csv(args.out_examples, index=False)
    print(f"gravado: {args.out_summary} e {args.out_examples}")
