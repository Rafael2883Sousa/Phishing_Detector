from email.parser import Parser
from email.policy import default

def parse_headers(raw: str) -> dict:
    msg = Parser(policy=default).parsestr(raw or "")
    from_ = msg.get("From", "")
    reply_to = msg.get("Reply-To", "")
    return_path = msg.get("Return-Path", "")
    auth = msg.get("Authentication-Results", "") or ""
    return {
        "from_reply_mismatch": (from_ and reply_to and (from_.split("@")[-1] != reply_to.split("@")[-1])),
        "from_return_mismatch": (from_ and return_path and (from_.split("@")[-1] != return_path.split("@")[-1])),
        "auth_spf_fail": ("spf=fail" in auth.lower()),
        "auth_dkim_fail": ("dkim=fail" in auth.lower()),
        "auth_dmarc_fail": ("dmarc=fail" in auth.lower()),
        "auth_any_none": any(tok in auth.lower() for tok in ["spf=none","dkim=none","dmarc=none"]),
    }
