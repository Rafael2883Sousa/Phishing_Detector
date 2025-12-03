# tools/augment_synthetic_modalities.py
#
# Adds synthetic headers_raw and html columns to data/processed/emails_full.csv
# to enable E2 (headers) and E3 (HTML) experiments.

import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "emails_full.csv"

URL_REGEX = re.compile(r"https?://\S+", flags=re.IGNORECASE)

LEGIT_DOMAINS = ["enron.com", "corp.com", "internal.net"]
ATTACKER_DOMAINS = ["evil.com", "phish.com", "secure-update.com"]

def extract_first_url(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    m = URL_REGEX.search(text)
    return m.group(0) if m else None


def build_synthetic_headers(label: str, url: str | None) -> str:
    """
    Build a synthetic headers_raw string that will trigger parse_headers()
    in a way correlated with label (ham/phishing).
    """
    # Choose base domains
    from_domain = random.choice(LEGIT_DOMAINS)
    reply_domain = from_domain
    return_domain = from_domain

    # Auth results defaults
    spf = "pass"
    dkim = "pass"
    dmarc = "pass"

    if label == "ham":
        if random.random() < 0.1:
            spf = "none"
        if random.random() < 0.05:
            dkim = "fail"
        if random.random() < 0.05:
            dmarc = "fail"
        if random.random() < 0.05:
            reply_domain = random.choice(ATTACKER_DOMAINS)
        if random.random() < 0.05:
            return_domain = random.choice(ATTACKER_DOMAINS)
    else:  # phishing
        
        if random.random() < 0.6:
            spf = "fail"
        elif random.random() < 0.3:
            spf = "none"

        if random.random() < 0.5:
            dkim = "fail"
        if random.random() < 0.5:
            dmarc = "fail"
        elif random.random() < 0.3:
            dmarc = "none"

        # Domain mismatches more likely
        if random.random() < 0.6:
            reply_domain = random.choice(ATTACKER_DOMAINS)
        if random.random() < 0.5:
            return_domain = random.choice(ATTACKER_DOMAINS)

    # If URL exists and label is phishing, try to align attacker domain with URL host
    if label == "phishing" and url:
        try:
            host = re.sub(r"^https?://", "", url)
            host = host.split("/")[0]
            # crude host -> domain extraction
            parts = host.split(".")
            if len(parts) >= 2:
                attack_dom = ".".join(parts[-2:])
                reply_domain = attack_dom
                return_domain = attack_dom
        except Exception:
            pass

    from_addr = f"alerts@{from_domain}"
    reply_addr = f"support@{reply_domain}"
    return_addr = f"bounce@{return_domain}"

    auth_results = f"Authentication-Results: spf={spf} dkim={dkim} dmarc={dmarc}"

    headers = (
        f"From: {from_addr}\n"
        f"Reply-To: {reply_addr}\n"
        f"Return-Path: {return_addr}\n"
        f"{auth_results}\n"
    )
    return headers


def build_synthetic_html(label: str, url: str | None) -> str:
    """
    Build synthetic HTML with an <a> tag, to trigger anchor_mismatch(html)
    in html_url.py. If no URL, return empty string.
    """
    if not url:
        return ""

    # Extract host for building anchor text
    host = re.sub(r"^https?://", "", url)
    host = host.split("/")[0]

    # Default: no mismatch, anchor text contains host
    anchor_text = f"Click here to visit {host}"

    mismatch = False
    if label == "ham":
        # Rare mismatch
        mismatch = random.random() < 0.1
    else:
        # Frequent mismatch for phishing
        mismatch = random.random() < 0.7

    if mismatch:
        # Use a generic "bank.com" or "secure portal" text that does not contain host
        options = [
            "Click here to access bank.com",
            "Secure portal login",
            "Update your account at bank.com",
        ]
        anchor_text = random.choice(options)

    html = f'<html><body><p>Notification:</p><a href="{url}">{anchor_text}</a></body></html>'
    return html


def main():
    df = pd.read_csv(DATA_PATH)

    required_cols = {"id", "label", "subject", "body", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in emails_full.csv: {missing}")

    has_headers = "headers_raw" in df.columns
    has_html = "html" in df.columns
    if has_headers or has_html:
        print("WARNING: columns headers_raw/html already exist and will be overwritten.")

    headers_list = []
    html_list = []

    for _, row in df.iterrows():
        label = row["label"]
        body = row["body"] if isinstance(row["body"], str) else ""
        url = extract_first_url(body)

        headers_raw = build_synthetic_headers(label=label, url=url)
        html = build_synthetic_html(label=label, url=url)

        headers_list.append(headers_raw)
        html_list.append(html)

    df["headers_raw"] = headers_list
    df["html"] = html_list

    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic headers_raw and html added to {DATA_PATH}")
    print(df[["label", "headers_raw", "html"]].head())


if __name__ == "__main__":
    main()
