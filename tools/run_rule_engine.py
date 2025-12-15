"""
Run rule engine on a CSV of test samples or a single JSON sample.
Usage:
    python tools/run_rule_engine.py --csv data/processed/emails_full.csv --out outputs/rule_matches.csv
    python tools/run_rule_engine.py --sample '{"id":1,"subject":"...","body":"..."}'
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from src.rules.engine import load_rule_engine

def run_on_csv(csv_path: Path, out_path: Path, cfg: str | None = None):
    df = pd.read_csv(csv_path)
    engine = load_rule_engine(cfg)
    rows = []
    for _, r in df.iterrows():
        sample = {
            "id": int(r["id"]),
            "subject": r.get("subject", ""),
            "body": r.get("body", ""),
            "urls": json.loads(r["urls"]) if "urls" in r and pd.notna(r["urls"]) else None,
            "headers_raw": r.get("headers_raw", ""),
            "html": r.get("html", ""),
        }
        reasons, details = engine.evaluate(sample)
        rows.append({"id": sample["id"], "reasons": json.dumps(reasons), "details": json.dumps(details)})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")

def run_on_sample(sample_json: str, cfg: str | None = None):
    sample = json.loads(sample_json)
    engine = load_rule_engine(cfg)
    reasons, details = engine.evaluate(sample)
    print(json.dumps({"reasons": reasons, "details": details}, indent=2, ensure_ascii=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/test_rule_matches.csv"))
    p.add_argument("--sample", type=str)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    if args.csv:
        run_on_csv(args.csv, args.out, args.config)
    elif args.sample:
        run_on_sample(args.sample, args.config)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
