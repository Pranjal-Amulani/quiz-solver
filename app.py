# app.py - Production-ready quiz solver (Playwright + heuristics + local file support)

import os
import re
import json
import time
import base64
import logging
from urllib.parse import urlparse, urljoin, unquote

from flask import Flask, request, jsonify, abort
import requests

# Playwright (optional if installed)
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

from bs4 import BeautifulSoup
import io
import pandas as pd
import pdfplumber

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
EXPECTED_SECRET = os.environ.get("QUIZ_SECRET", "s3cr3t-quiz-$(Rk8)-2025")
USER_EMAIL = os.environ.get("QUIZ_EMAIL", "your.email@example.com")
BROWSER_TIMEOUT = int(os.environ.get("BROWSER_TIMEOUT_S", "60"))  # seconds
QUIZ_TIMEOUT_S = 180  # 3 minutes allowed for a quiz chain


# -------------------------
# Utilities
# -------------------------
def json_or_400(req):
    if not req.is_json:
        abort(400, description="Invalid JSON")
    try:
        return req.get_json()
    except Exception:
        abort(400, description="Invalid JSON payload")


def write_debug_html(content, fname="rendered_debug.html"):
    """Write debug HTML into CWD so you can open in VSCode."""
    try:
        path = os.path.join(os.getcwd(), fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info("Rendered HTML saved to: %s", path)
    except Exception:
        logging.exception("Failed to write debug HTML")


def windows_path_from_file_url(file_url):
    """
    Convert 'file:///D:/path/to/file' or 'file://localhost/C:/path' to Windows path.
    Handles leading slash that URL parsers include.
    """
    p = urlparse(file_url).path  # often '/D:/...'
    if p.startswith("/") and len(p) > 2 and p[2] == ":":
        p = p[1:]  # remove leading slash -> 'D:/...'
    # unquote percent-encoded spaces etc
    return unquote(p)


# -------------------------
# HTML parsing / heuristics
# -------------------------
def try_parse_json_text(text):
    """Try parse JSON directly or after base64 decode."""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        cleaned = re.sub(r"\s+", "", text)
        decoded = base64.b64decode(cleaned)
        return json.loads(decoded.decode("utf-8", errors="ignore"))
    except Exception:
        pass
    return None


def find_submit_info(html_text, base_url):
    """
    Heuristic extraction:
    - <pre> with plain JSON or base64 JSON
    - <script> blobs with JSON or base64
    - anchor hrefs containing 'submit'
    - regex absolute submit links
    - 'The code word is: <word>'
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # 1) <pre>
    pre = soup.find("pre")
    if pre:
        parsed = try_parse_json_text(pre.get_text())
        if parsed:
            return {"raw": parsed}

    # 2) script tags: search for long base64 or object literals
    for script in soup.find_all("script"):
        s = (script.string or "") or script.get_text() or ""
        # base64-like segments
        for m in re.finditer(r'([A-Za-z0-9+/]{40,}={0,2})', s):
            cand = m.group(1)
            try:
                decoded = base64.b64decode(cand)
                parsed = try_parse_json_text(decoded.decode("utf-8", errors="ignore"))
                if parsed:
                    return {"raw": parsed}
            except Exception:
                pass
        # object literal in script
        mjson = re.search(r'(\{[\s\S]{20,}\})', s)
        if mjson:
            parsed = try_parse_json_text(mjson.group(1))
            if parsed:
                return {"raw": parsed}

    # 3) submit anchors
    for a in soup.find_all("a", href=True):
        if "submit" in a["href"].lower():
            return {"submit_url": urljoin(base_url, a["href"])}

    # 4) absolute submit url
    m = re.search(r"https?://[^\s'\"<>]*submit[^\s'\"<>]*", html_text, re.I)
    if m:
        return {"submit_url": m.group(0)}

    # 5) code word
    m2 = re.search(r"The code word is: ?([A-Za-z0-9_-]+)", html_text, re.I)
    if m2:
        return {"code_word": m2.group(1)}

    return {"html": html_text}


# -------------------------
# PDF parsing helper
# -------------------------
def parse_pdf_and_compute_sum(pdf_bytes, column_name="value", page_number=2):
    s = io.BytesIO(pdf_bytes)
    with pdfplumber.open(s) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            raise ValueError("Invalid page number")
        page = pdf.pages[page_number - 1]
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table[1:], columns=table[0])
            for col in df.columns:
                if col and column_name.lower() in col.lower():
                    series = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True),
                                           errors='coerce')
                    return float(series.sum(skipna=True))
    raise ValueError("Could not find column to sum in PDF")


# -------------------------
# Render page with Playwright (if available)
# -------------------------
def render_page_playwright(url, timeout_s=BROWSER_TIMEOUT):
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not available in this Python environment")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout_s * 1000)
        except PlaywrightTimeoutError:
            logging.info("networkidle timeout; doing plain goto")
            page.goto(url, timeout=timeout_s * 1000)
        # wait briefly for <pre> injected by some pages
        try:
            page.wait_for_selector("pre", timeout=15000)
            logging.info("<pre> appeared in DOM")
        except Exception:
            logging.debug("No <pre> within wait")
        content = page.content()
        final_url = page.url
        context.close()
        browser.close()
        return content, final_url


# -------------------------
# Solve & submit single step
# -------------------------
def solve_single(task_url, session):
    """
    Given a task_url (http/https or file:///), return (submit_url, answer).
    For file:// paths, we read the file locally and attempt to parse.
    """
    parsed = urlparse(task_url)
    scheme = parsed.scheme.lower()

    html = None
    final_url = task_url

    if scheme == "file":
        # Convert to local path and read content
        local_path = windows_path_from_file_url(task_url)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        # if PDF:
        if local_path.lower().endswith(".pdf"):
            pdf_bytes = open(local_path, "rb").read()
            # No submit_url present in file - caller must provide submit or page includes submit link in html
            # We'll attempt to compute a typical 'sum value on page 2' if asked externally; otherwise return no submit
            try:
                total = parse_pdf_and_compute_sum(pdf_bytes, column_name="value", page_number=2)
                return None, total
            except Exception:
                return None, None
        # if HTML:
        if local_path.lower().endswith((".htm", ".html")):
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
                final_url = task_url
        # if image:
        if local_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            if OCR_AVAILABLE:
                img_bytes = open(local_path, "rb").read()
                text = ocr_image_bytes(img_bytes)
                # very naive: look for "The code word is: <word>"
                m = re.search(r"The code word is: ?([A-Za-z0-9_-]+)", text, re.I)
                if m:
                    return None, m.group(1)
            return None, None

    else:
        # HTTP(S) - prefer Playwright for JS rendering
        try:
            if PLAYWRIGHT_AVAILABLE:
                html, final_url = render_page_playwright(task_url)
            else:
                r = session.get(task_url, timeout=20)
                r.raise_for_status()
                html = r.text
                final_url = r.url
        except Exception:
            # fallback to requests if Playwright failed
            try:
                r = session.get(task_url, timeout=20)
                r.raise_for_status()
                html = r.text
                final_url = r.url
            except Exception as e:
                logging.exception("Failed to load URL: %s", e)
                raise

    # Save debug HTML so you can inspect
    if html:
        try:
            write_debug_html(html)
        except Exception:
            pass

    info = find_submit_info(html, final_url)
    logging.info("find_submit_info returned keys: %s", list(info.keys()))

    # If raw JSON embedded
    if "raw" in info:
        data = info["raw"]
        # prefer explicit submit in JSON
        submit_url = data.get("submit") or data.get("submit_url")
        answer = data.get("answer")
        # If JSON includes a URL to a file to download, and text instructs to sum, attempt that
        data_url = data.get("url")
        if data_url and (("sum" in json.dumps(data).lower()) or re.search(r"sum of the .*value", html or "", re.I)):
            file_url = urljoin(final_url, data_url)
            file_bytes = session.get(file_url, timeout=30).content
            try:
                total = parse_pdf_and_compute_sum(file_bytes, column_name="value", page_number=2)
                answer = total
            except Exception:
                pass
        return submit_url, answer

    # If code_word found
    if "code_word" in info:
        return final_url, info["code_word"]

    # If submit link found, maybe it's a PDF task
    submit_url = info.get("submit_url")
    # Look for PDF links and question asking for sum
    soup = BeautifulSoup(html or "", "html.parser")
    pdf_href = None
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".pdf"):
            pdf_href = urljoin(final_url, a["href"])
            break
    if pdf_href and re.search(r"sum of the .*value", html or "", re.I):
        try:
            pdf_bytes = session.get(pdf_href, timeout=30).content
            total = parse_pdf_and_compute_sum(pdf_bytes, column_name="value", page_number=2)
            return submit_url, total
        except Exception:
            logging.exception("PDF parse failed")

    # Look for Answer: pattern
    m = re.search(r'Answer[:\s]+([A-Za-z0-9\-\_ ]+)', html or "", re.I)
    if m:
        return submit_url or final_url, m.group(1).strip()

    # fallback
    return submit_url, None


# -------------------------
# /api/quiz endpoint
# -------------------------
@app.route("/api/quiz", methods=["POST"])
def quiz_webhook():
    payload = json_or_400(request)
    email = payload.get("email")
    secret = payload.get("secret")
    first_url = payload.get("url")
    if not (email and secret and first_url):
        abort(400, description="Missing email, secret, or url")

    if secret != EXPECTED_SECRET:
        return jsonify({"ok": False, "error": "invalid secret"}), 403

    start_time = time.time()
    session = requests.Session()
    current_url = first_url
    submit_log = []
    last_submit_response = None

    while True:
        if time.time() - start_time > QUIZ_TIMEOUT_S:
            logging.warning("Quiz time budget exceeded")
            break

        try:
            submit_url, answer = solve_single(current_url, session)
        except Exception as e:
            logging.exception("Failed to process URL %s: %s", current_url, e)
            submit_log.append({"url": current_url, "error": str(e)})
            break

        logging.info("Determined submit_url: %s answer: %s", submit_url, str(answer))

        if not submit_url:
            submit_log.append({"url": current_url, "message": "No submit URL discovered"})
            break

        post_payload = {"email": email, "secret": secret, "url": current_url}
        if answer is not None:
            post_payload["answer"] = answer
        # ensure under 1MB
        body = json.dumps(post_payload)
        if len(body.encode("utf-8")) > 1024 * 1024:
            submit_log.append({"url": current_url, "error": "payload too large"})
            break

        try:
            r = session.post(submit_url, json=post_payload, timeout=30)
            try:
                r.raise_for_status()
            except Exception:
                logging.exception("Submit POST returned non-2xx for %s", submit_url)
            try:
                resp_json = r.json()
            except Exception:
                resp_json = {"status": r.status_code, "text": r.text}
            submit_log.append({"url": current_url, "submit_url": submit_url, "response": resp_json})
            last_submit_response = resp_json
            # follow next URL if provided
            if isinstance(resp_json, dict) and resp_json.get("url"):
                current_url = resp_json["url"]
                continue
            # else done
            break
        except Exception as e:
            logging.exception("Submit failed: %s", e)
            submit_log.append({"url": current_url, "submit_url": submit_url, "error": str(e)})
            break

    result = {"ok": True, "summary": submit_log, "elapsed_s": time.time() - start_time, "last_submit": last_submit_response}
    return jsonify(result), 200


if __name__ == "__main__":
    # Local dev server entrypoint. In production use gunicorn or host's recommended runner.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
