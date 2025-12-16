from email.parser import Parser
from email.policy import default
from email.utils import getaddresses

def parse_headers(raw: str) -> dict:
    msg = Parser(policy=default).parsestr(raw or "")

    from_addrs = getaddresses(msg.get_all("From", []))
    reply_addrs = getaddresses(msg.get_all("Reply-To", []))
    return_addrs = getaddresses(msg.get_all("Return-Path", []))

    auth = msg.get("Authentication-Results", "") or ""

    return {
        
        "from_domains": [a.split("@")[-1].lower() for _, a in from_addrs if "@" in a],
        "reply_domains": [a.split("@")[-1].lower() for _, a in reply_addrs if "@" in a],
        "return_domains": [a.split("@")[-1].lower() for _, a in return_addrs if "@" in a],

        "auth_spf_fail": ("spf=fail" in auth.lower()),
        "auth_dkim_fail": ("dkim=fail" in auth.lower()),
        "auth_dmarc_fail": ("dmarc=fail" in auth.lower()),
        "auth_any_none": any(tok in auth.lower() for tok in ["spf=none","dkim=none","dmarc=none"]),
    }
