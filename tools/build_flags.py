# -*- coding: utf-8 -*-
# Derive robustness flags from raw test data.
import argparse, re, json, pandas as pd
from urllib.parse import urlsplit
SHORTENERS = {"bit.ly","tinyurl.com","goo.gl","ow.ly","t.co","is.gd","buff.ly","cutt.ly","rebrand.ly","lnkd.in","s.id"}
REDIRECTORS = {"l.facebook.com","l.instagram.com","nav.smartscreen.microsoft.com","urldefense.proofpoint.com","secure-web.cisco.com"}
PTBR_MARKERS = {"você","fatura","boleto","cadastre-se","celular","cpf","pix","favorecido","transferência","cartão de crédito"}
HOMOGLYPH_HINT = r"[^\x00-\x7F]"  

def extract_urls(text):
    # regex simples para URL
    rgx = re.compile(r"https?://[^\s)>\]}\"']+", re.I)
    return rgx.findall(text or "")

def domain(u):
    try:
        return urlsplit(u).netloc.lower()
    except Exception:
        return ""

def has_shortener(urls):
    return any(any(d.endswith(s) or d==s for s in SHORTENERS) for d in map(domain, urls))

def has_redirector(urls):
    return any(domain(u) in REDIRECTORS for u in urls)

def unicode_homoglyphs(text):
    return bool(re.search(HOMOGLYPH_HINT, text or ""))

def ptbr_likely(text):
    t = (text or "").lower()
    return any(w in t for w in PTBR_MARKERS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_test", required=True)       # ex.: data/processed/test_email.csv
    ap.add_argument("--out", required=True)            # ex.: outputs/test_flags.csv
    ap.add_argument("--mode", choices=["email","sms"], default="email")
    ap.add_argument("--subject_col", default="subject")
    ap.add_argument("--body_col", default="body")
    ap.add_argument("--headers_col", default="headers_json")  # pode não existir
    args = ap.parse_args()

    df = pd.read_csv(args.csv_test)
    subj = df.get(args.subject_col, pd.Series([""]*len(df))).astype(str)
    body = df.get(args.body_col, pd.Series([""]*len(df))).astype(str)
    text = (subj + " " + body).str.strip()

    urls = text.apply(extract_urls)
    df["has_url"] = urls.apply(lambda v: len(v)>0)
    df["shortened_url"] = urls.apply(has_shortener)
    df["redirector_url"] = urls.apply(has_redirector)
    df["unicode_homoglyph"] = text.apply(unicode_homoglyphs)
    df["ptbr_likely"] = text.apply(ptbr_likely)
    df["no_url"] = ~df["has_url"]

    # headers ausentes/incompletos
    if args.headers_col in df.columns:
        def hdr_missing(v):
            try:
                h = json.loads(v) if isinstance(v, str) and v.strip().startswith("{") else {}
            except Exception:
                h = {}
            # consideramos "ausente" se não há SPF/DKIM/DMARC avaliáveis
            keys = [k.lower() for k in h.keys()] if isinstance(h, dict) else []
            return not any(x in keys for x in ("spf","dkim","dmarc","authentication-results"))
        df["headers_missing"] = df[args.headers_col].apply(hdr_missing)
    else:
        df["headers_missing"] = True  # se não tens cabeçalhos no dataset, marca como ausentes

    df_out = df[["id","has_url","shortened_url","redirector_url","unicode_homoglyph","ptbr_likely","no_url","headers_missing"]]
    df_out.to_csv(args.out, index=False)
    print(f"gravado: {args.out} ({len(df_out)})")

if __name__ == "__main__":
    main()
