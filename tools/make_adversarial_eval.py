# Gera variações adversariais a partir do CSV de teste para forçar FP/FN plausíveis.
import argparse, pandas as pd, numpy as np, json, re, random
from urllib.parse import urlsplit

RNG = np.random.default_rng(42)

PTBR_INJ = [
 "Você precisa atualizar seu cadastro", "fatura em atraso", "boleto disponível",
 "celular cadastrado", "CPF requerido", "pix com desconto", "cartão de crédito bloqueado"
]
NEWSLETTER_WRAP = [
 "Oferta exclusiva termina hoje", "Cupom válido até às 23:59",
 "Últimas unidades com 30% OFF", "Assine e receba vantagens"
]
SHORTENERS = ["https://bit.ly/abc", "https://t.co/xyz", "https://tinyurl.com/zz1"]
REDIR = "https://l.facebook.com/l.php?u={}"

URL_RGX = re.compile(r"https?://[^\s)>\]}\"']+", re.I)

def first_url(text):
    m = URL_RGX.search(text or "")
    return m.group(0) if m else None

def replace_with_shortener(text):
    u = first_url(text)
    if not u: return text
    return text.replace(u, RNG.choice(SHORTENERS))

def wrap_with_redirector(text):
    u = first_url(text)
    if not u: return text
    return text.replace(u, REDIR.format(u))

def remove_urls(text):
    return URL_RGX.sub("", text or "")

def add_ptbr(text):
    return (text or "") + " " + RNG.choice(PTBR_INJ)

def add_newsletter(subject, body):
    s = f"[{RNG.choice(NEWSLETTER_WRAP)}] {subject or ''}".strip()
    b = (body or "") + " Cancelar subscrição | Gerir preferências"
    return s, b

def make_headers(auth_ok=True):
    if auth_ok:
        return json.dumps({"spf":"pass","dkim":"pass","dmarc":"pass","authentication-results":"spf=pass dkim=pass dmarc=pass"})
    return json.dumps({"authentication-results":"none"})

def build_text_email(row):
    subj = str(row.get("subject","") or "")
    body = str(row.get("body","") or "")
    return subj, body

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_test", required=True)      
    ap.add_argument("--out_csv", required=True)       
    ap.add_argument("--target_n", type=int, default=600)
    args = ap.parse_args()

    base = pd.read_csv(args.csv_test)
    assert "label" in base.columns, "espera coluna 'label' com phishing/legit ou 1/0/ham/spam"
    # normaliza rótulo
    ymap = {"phishing":1,"spam":1,"legit":0,"ham":0,"1":1,"0":0}
    base["y"] = base["label"].astype(str).str.lower().map(ymap).astype(int)

    rows = []
    # copia base
    for _,r in base.iterrows():
        rows.append(r.to_dict())

    # sintetiza variações
    while len(rows) < args.target_n:
        r = base.sample(1, random_state=int(RNG.integers(0,1e9))).iloc[0].copy()
        subj, body = build_text_email(r)
        y = int(r["y"])

        mode = RNG.choice(["FN_redirect","FN_semurl","FP_news","FP_ptbr","FP_headerok"])
        headers_json = r.get("headers_json", "")

        if mode=="FN_redirect" and y==1:
            # phish com redirect/encurtador → tende a cair para FN
            body2 = wrap_with_redirector(body)
            if body2==body: body2 = replace_with_shortener(body)
            r["body"] = body2
        elif mode=="FN_semurl" and y==1:
            # phish sem URL (pede resposta) → tende a FN
            r["body"] = remove_urls(body) + " Responda confirmando seus dados."
        elif mode=="FP_news" and y==0:
            # legítimo com aparência promocional → tende a FP
            s2, b2 = add_newsletter(subj, body)
            r["subject"], r["body"] = s2, b2
        elif mode=="FP_ptbr" and y==0:
            # legítimo com marcadores PT-BR → tende a FP
            r["body"] = add_ptbr(body)
        elif mode=="FP_headerok" and y==0:
            # legítimo com headers autenticados explícitos → reforça FP_header_ok se conteúdo soar suspeito
            r["headers_json"] = make_headers(auth_ok=True)
            r["subject"] = f"Ação necessária: {subj}"
            r["body"] = (body or "") + " Atualize sua senha imediatamente."
        else:
            # se incompatível com o rótulo, apenas duplica com pequeno ruído lexical
            r["body"] = (body or "") + " Obrigado."

        # assegura campos mínimos
        r["subject"] = str(r.get("subject","") or "")
        r["body"] = str(r.get("body","") or "")
        if "headers_json" not in r or pd.isna(r["headers_json"]):
            r["headers_json"] = make_headers(auth_ok=False)

        rows.append(r.to_dict())

    out = pd.DataFrame(rows).reset_index(drop=True)
    # renumera id sequencial para merge posterior
    out["id"] = np.arange(len(out))
    out.to_csv(args.out_csv, index=False)
    print(f"gravado: {args.out_csv} ({len(out)} linhas)")
