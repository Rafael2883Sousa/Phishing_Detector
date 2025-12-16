"""
Rule Engine for the phishing detection prototype.

- Lightweight, deterministic rules that complement the ML model.
- Returns a list of triggered rule keys ("reasons") and
  a structured explanation for each rule (for audit / UI).
- Reuses functions from src.features when available, else uses local fallbacks.
"""

from typing import Dict, Any, List, Tuple
import re
import logging
from importlib import util
from pathlib import Path
import yaml

logger = logging.getLogger("rule_engine")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT.joinpath("configs", "rules.yaml")

# Try to import feature helpers from src.features when present
def _import_feature(name: str):
    spec = util.find_spec(f"src.features.{name}")
    if spec:
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module) 
        return module
    return None

_url_signals = _import_feature("url_signals")
_headers = _import_feature("headers")
_html_url = _import_feature("html_url")

# Local fallbacks
URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)

def extract_urls(text: str) -> List[str]:
    if _url_signals and hasattr(_url_signals, "extract_urls"):
        try:
            return _url_signals.extract_urls(text)
        except Exception:
            pass
    return URL_RE.findall(text or "")

def has_at_in_url(url: str) -> bool:
    return "@" in url

def host_from_url(url: str) -> str:
    try:
        u = re.sub(r"^https?://", "", url)
        host = u.split("/", 1)[0]
        return host.lower()
    except Exception:
        return ""

# Default suspect TLDs
DEFAULT_SUSPECT_TLDS = {".tk", ".ml", ".cf", ".gq", ".zip", ".mov"}

def tld_of_host(host: str) -> str:
    parts = host.split(".")
    if len(parts) < 2:
        return ""
    return "." + parts[-1]

# Rule engine class
class RuleEngine:
    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or CONFIG_PATH
        self.config = self._load_config()
        self.suspect_tlds = set(self.config.get("suspect_tlds", DEFAULT_SUSPECT_TLDS))
        self.shortener_hosts = set(self.config.get("shorteners", []))
        self.redirector_indicators = set(self.config.get("redirectors", []))
        self.ptbr_markers = set(self.config.get("ptbr_markers", []))

    def run(self, sample: dict) -> dict:
        """
        Canonical entrypoint for the Rule Engine.
        Separates accusatory vs neutral rules.
        """
        reasons, details = self.evaluate(sample)

        ACCUSATORY_RULES = {
            "has_at",
            "tld_suspect",
            "anchor_href_mismatch",
            "auth_spf_fail",
            "auth_dkim_fail",
            "auth_dmarc_fail",
            "from_reply_mismatch",
            "from_return_mismatch",
            "redirector_url",
            "shortened_url",
            "confusable",
        }

        accusatory_hits = [r for r in reasons if r in ACCUSATORY_RULES]
        neutral_hits = [r for r in reasons if r not in ACCUSATORY_RULES]

        risk_score = min(len(accusatory_hits) / 3.0, 1.0)

        return {
            "reasons": reasons,
            "accusatory_hits": accusatory_hits,
            "neutral_hits": neutral_hits,
            "risk_score": risk_score,
            "details": details,
        }

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    return cfg or {}
            except Exception as e:
                logger.warning("Failed to parse rules config: %s", e)
                return {}
        return {}
    
    EMAIL_RE = re.compile(
        r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
    )

    def extract_email(value: str) -> str:
        m = EMAIL_RE.search(value or "")
        return m.group(1).lower() if m else ""

    def evaluate(self, sample: Dict[str, Any]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Evaluate all rules against a single sample.

        sample expected keys: id, subject, body, urls (optional list), headers_raw, html

        Returns:
            reasons: list of reason keys (strings)
            details: mapping reason->explanation dict for auditing
        """
        reasons: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        # prepare canonical fields
        subject = (sample.get("subject") or "") or ""
        body = (sample.get("body") or "") or ""
        text = (subject + " " + body).strip()
        urls = sample.get("urls")
        html = (sample.get("html") or "") or ""
        if urls is None:
            urls = extract_urls(text)
        if html:
            urls += extract_urls(html)
        headers_raw = (sample.get("headers_raw") or "") or ""
       

        # Rule: has_at_in_url
        for u in urls:
            if has_at_in_url(u):
                key = "has_at"
                reasons.append(key)
                details.setdefault(key, {"matches": []})["matches"].append(u)

        # Rule: tld_suspect
        for u in urls:
            host = host_from_url(u)
            tld = tld_of_host(host)
            if tld and tld in self.suspect_tlds:
                key = "tld_suspect"
                reasons.append(key)
                details.setdefault(key, {"hosts": []})["hosts"].append(host)

        # Rule: confusable (basic unicode non-ascii presence in host)
        for u in urls:
            host = host_from_url(u)
            if any(ord(ch) > 127 for ch in host):
                key = "confusable"
                reasons.append(key)
                details.setdefault(key, {"hosts": []})["hosts"].append(host)

        # Rule: shortened_url
        if self.shortener_hosts:
            for u in urls:
                host = host_from_url(u)
                for s in self.shortener_hosts:
                    if host.startswith(s):
                        key = "shortened_url"
                        reasons.append(key)
                        details.setdefault(key, {"hosts": []})["hosts"].append(host)
                        break

        # Rule: redirector_url (look for known redirectors or safe links)
        if self.redirector_indicators:
            for u in urls:
                host = host_from_url(u)
                for r in self.redirector_indicators:
                    if r in host:
                        key = "redirector_url"
                        reasons.append(key)
                        details.setdefault(key, {"hosts": []})["hosts"].append(host)
                        break

        # Rule: ptbr_likely - language marker tokens
        lower_text = text.lower()
        if any(tok in lower_text for tok in self.ptbr_markers):
            key = "ptbr_likely"
            reasons.append(key)
            details.setdefault(key, {})["evidence"] = [tok for tok in self.ptbr_markers if tok in lower_text]

        # Rule: no_url
        if len(urls) == 0:
            key = "no_url"
            reasons.append(key)
            details.setdefault(key, {})["note"] = "no URLs extracted from subject/body/html"

        # Rule: anchor_href_mismatch (use html_url if present)
        if _html_url and hasattr(_html_url, "anchor_mismatch"):
            try:
                mismatch = _html_url.anchor_mismatch(html)
                if mismatch:
                    key = "anchor_href_mismatch"
                    reasons.append(key)
                    details.setdefault(key, {})["note"] = "anchor text and href mismatch detected"
            except Exception:
                # fallback: basic heuristic - check for 'bank.com' presence vs href host
                pass
        else:
            # simple fallback heuristic: look for pattern "bank" in text and href to other host
            # naive, but safe as fallback
            anchors = re.findall(r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", html, flags=re.IGNORECASE)
            for href, anchor_text in anchors:
                anchor_lower = anchor_text.lower()
                host = host_from_url(href)
                if host and (host not in anchor_lower):
                    key = "anchor_href_mismatch"
                    reasons.append(key)
                    details.setdefault(key, {"examples": []})["examples"].append({"href": href, "anchor": anchor_text})

        # Rule: header auth failures using headers parser if available
        if _headers and hasattr(_headers, "parse_headers"):
            try:
                parsed = _headers.parse_headers(headers_raw)
                # parsed expected keys: auth_spf_fail, auth_dkim_fail, auth_dmarc_fail, from_reply_mismatch, from_return_mismatch
                for k in ("auth_spf_fail", "auth_dkim_fail", "auth_dmarc_fail"):
                    if parsed.get(k):
                        reasons.append(k)
                        details.setdefault(k, {})["value"] = True
                for k in ("from_reply_mismatch", "from_return_mismatch"):
                    if parsed.get(k):
                        reasons.append(k)
                        details.setdefault(k, {})["value"] = True
            except Exception:
                logger.debug("headers.parse_headers failed", exc_info=True)
        else:
            # Basic fallback checks in headers_raw
            if headers_raw:
                if "spf=fail" in headers_raw.lower():
                    reasons.append("auth_spf_fail")
                    details.setdefault("auth_spf_fail", {})["evidence"] = "spf=fail in headers_raw"
                if "dkim=fail" in headers_raw.lower():
                    reasons.append("auth_dkim_fail")
                    details.setdefault("auth_dkim_fail", {})["evidence"] = "dkim=fail in headers_raw"
                if "dmarc=fail" in headers_raw.lower():
                    reasons.append("auth_dmarc_fail")
                    details.setdefault("auth_dmarc_fail", {})["evidence"] = "dmarc=fail in headers_raw"
                # From/Reply mismatch naive
                if "from:" in headers_raw.lower() and "reply-to:" in headers_raw.lower():
                    try:
                        from_match = re.search(r"From:\s*([^\n\r]+)", headers_raw, flags=re.IGNORECASE)
                        reply_match = re.search(r"Reply-To:\s*([^\n\r]+)", headers_raw, flags=re.IGNORECASE)

                        if from_match and reply_match:
                            from_email = extract_email(from_match.group(1))
                            reply_email = extract_email(reply_match.group(1))

                            if from_email and reply_email:
                                from_domain = from_email.split("@")[1]
                                reply_domain = reply_email.split("@")[1]

                                if from_domain != reply_domain:
                                    reasons.append("from_reply_mismatch")
                                    details.setdefault("from_reply_mismatch", {})["value"] = {
                                        "from": from_email,
                                        "reply_to": reply_email
                                    }
                    except Exception:
                        pass

        # Deduplicate reasons while preserving order
        seen = set()
        unique_reasons = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                unique_reasons.append(r)

        return unique_reasons, details


# Convenience factory function
def load_rule_engine(cfg_path: str | None = None) -> RuleEngine:
    return RuleEngine(config_path=Path(cfg_path) if cfg_path else None)
