import unicodedata
from urllib.parse import urlparse

SUSPECT_TLDS = {"zip","mov","gq","tk","ml","cf"}

def normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def _count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s or "")

def _confusable_alert(host: str) -> bool:
    scripts = set(unicodedata.name(c).split()[0] for c in host if c.isalpha())
    return len(scripts) > 1

def url_features(u: str) -> dict:
    u = normalize_unicode((u or "").strip())
    parsed = urlparse(u if "://" in u else "http://" + u)
    host = parsed.hostname or ""
    path = parsed.path or ""
    subdomains = host.split(".")[:-2] if host.count(".") >= 2 else []
    return {
        "len_url": len(u),
        "len_host": len(host),
        "len_path": len(path),
        "num_dots": u.count("."),
        "num_digits": _count_digits(u),
        "num_subdomains": len(subdomains),
        "has_at": "@" in u,
        "has_double_slash": "//" in path,
        "tld_suspect": host.split(".")[-1] in SUSPECT_TLDS if "." in host else False,
        "confusable": _confusable_alert(host),
    }
