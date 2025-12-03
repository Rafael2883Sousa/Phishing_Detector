# tools/build_test_flags.py
#
# Build robustness flags for the TEST SET used in E3.
# Input:
#   data/processed/emails_full.csv
#   outputs/E3_test_predictions_email.csv
# Output:
#   outputs/test_flags.csv
#
# Columns:
#   id, label,
#   unicode_homoglyph, shortened_url, redirector_url,
#   ptbr_likely, no_url, headers_missing

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMAILS_PATH = ROOT / "data" / "processed" / "emails_full.csv"
PRED_E3_PATH = ROOT / "outputs" / "E3_test_predictions_email.csv"
OUT_PATH = ROOT / "outputs" / "test_flags.csv"

# --- Config ---

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "buff.ly", "cutt.ly", "lnkd.in", "rb.gy"
}

REDIRECTORS = {
    "l.facebook.com",
    "l.instagram.com",
    "urldefense.proofpoint.com",
    "safelinks.protection.outlook.com",
    "click.mail",
    "click.email",
}

PTBR_MARKERS = {
    "você", "voce", "cpf", "boleto", "pix", "celular",
    "fatura", "segunda via", "atualizar cadastro",
}


def extract_urls(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return URL_RE.findall(text)


def get_host(url: str) -> str:
    try:
        u = re.sub(r"^https?://", "", url)
        host = u.split("/", 1)[0]
        return host.lower()
    except Exception:
        return ""


def has_shortener(urls: list[str]) -> bool:
    for u in urls:
        host = get_host(u)
        for s in SHORTENERS:
            if host.startswith(s):
                return True
    return False


def has_redirector(urls: list[str]) -> bool:
    for u in urls:
        host = get_host(u)
        for r in REDIRECTORS:
            if r in host:
                return True
    return False


def has_unicode_homoglyph(text: str) -> bool:
    if not isinstance(text, str):
        return False
    # True se existir pelo menos um caractere não-ASCII
    return any(ord(ch) > 127 for ch in text)


def is_ptbr_likely(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lower = text.lower()
    return any(tok in lower for tok in PTBR_MARKERS)


def headers_are_missing(headers_raw: str) -> bool:
    if not isinstance(headers_raw, str) or not headers_raw.strip():
        return True
    lower = headers_raw.lower()
    tokens = ["spf=", "dkim=", "dmarc="]
    return not any(t in lower for t in tokens)


def main():
    if not EMAILS_PATH.exists():
        raise FileNotFoundError(f"emails_full.csv não encontrado em {EMAILS_PATH}")
    if not PRED_E3_PATH.exists():
        raise FileNotFoundError(f"E3_test_predictions_email.csv não encontrado em {PRED_E3_PATH}")

    emails = pd.read_csv(EMAILS_PATH)
    preds = pd.read_csv(PRED_E3_PATH)

    required_emails = {"id", "label", "subject", "body"}
    missing = required_emails - set(emails.columns)
    if missing:
        raise ValueError(f"Colunas em falta em emails_full.csv: {missing}")

    if "headers_raw" not in emails.columns:
        emails["headers_raw"] = ""

    # ids que pertencem ao TEST SET de E3
    test_ids = set(preds["id"].tolist())

    # restringir emails ao test set
    df = emails[emails["id"].isin(test_ids)].copy()
    if df.empty:
        raise ValueError("Nenhuma linha de emails_full.csv corresponde aos ids de teste de E3.")

    flags_rows = []

    for _, row in df.iterrows():
        rid = row["id"]
        label = row["label"]
        subject = row.get("subject", "") or ""
        body = row.get("body", "") or ""
        headers_raw = row.get("headers_raw", "") or ""

        text = f"{subject} {body}"
        urls = extract_urls(text)

        row_flags = {
            "id": rid,
            "label": label,
            "unicode_homoglyph": has_unicode_homoglyph(text),
            "shortened_url": has_shortener(urls),
            "redirector_url": has_redirector(urls),
            "ptbr_likely": is_ptbr_likely(text),
            "no_url": len(urls) == 0,
            "headers_missing": headers_are_missing(headers_raw),
        }
        flags_rows.append(row_flags)

    out_df = pd.DataFrame(flags_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"test_flags.csv gravado em: {OUT_PATH}")
    print(out_df.head())
    print("Resumo por flag:")
    for col in ["unicode_homoglyph", "shortened_url", "redirector_url",
                "ptbr_likely", "no_url", "headers_missing"]:
        print(col, "=>", out_df[col].sum())


if __name__ == "__main__":
    main()
