from bs4 import BeautifulSoup
from urllib.parse import urlparse

def anchor_mismatch(html: str) -> bool:
    soup = BeautifulSoup(html or "", "lxml")
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        href = (a.get("href") or "").strip().lower()
        if not href or href.startswith("mailto:"):
            continue
        host = urlparse(href if "://" in href else "http://" + href).hostname or ""
        if host and text and host not in text:
            return True
    return False
