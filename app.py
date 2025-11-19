# app.py
# app.py
import os
import re
import json
import time
import base64
import logging
from urllib.parse import urljoin, urlparse

from flask import Flask, request, jsonify, abort
import requests

# Playwright for headless browser to execute JS pages
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# For PDF processing and data frames
import io
import pandas as pd
import pdfplumber  # pip install pdfplumber

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configure these from environment or hardcode for testing
EXPECTED_SECRET = os.environ.get("QUIZ_SECRET", "s3cr3t-quiz-$(Rk8)-2025")
USER_EMAIL = os.environ.get("QUIZ_EMAIL", "your.email@example.com")
BROWSER_TIMEOUT = int(os.environ.get("BROWSER_TIMEOUT_S", "60"))  # seconds for page load

def json_or_400(req):
    if not req.is_json:
        abort(400, description="Invalid JSON")
    try:
        return req.get_json()
    except Exception:
        abort(400, description="Invalid JSON payload")

def find_submit_info(html_text, base_url):
    """
    Try several heuristics to find the submit URL and what to answer.
    Return a dict containing keys: 'submit_url', 'answer_spec', 'hints'
    answer_spec could be a dict instructing how to compute the answer.
    """
    # Heuristic 1: Look for a JSON blob in <pre> or script tags (common in sample)
    m = re.search(r"<pre.*?>([\s\S]*?)</pre>", html_text, re.I)
    if m:
        try:
            decoded = m.group(1).strip()
            # Some pages base64-encode; attempt base64 decode if it looks like base64
            try:
                decoded_b = base64.b64decode(re.sub(r'\s+', '', decoded))
                decoded_text = decoded_b.decode('utf-8', errors='ignore')
                data = json.loads(decoded_text)
                return {"raw": data}
            except Exception:
                # maybe it's plain JSON
                data = json.loads(decoded)
                return {"raw": data}
        except Exception:
            pass

    # Heuristic 2: Look for "submit" urls in HTML
    m2 = re.search(r"https?://[^\s'\"<>]+/submit[^\s'\"<>]*", html_text)
    if m2:
        return {"submit_url": m2.group(0)}

    # Heuristic 3: Look for "The code word is" pattern (for tests)
    m3 = re.search(r"The code word is: ?([A-Za-z0-9_-]+)", html_text)
    if m3:
        return {"code_word": m3.group(1)}

    # No clear result
    return {"html": html_text}

def download_file(session, url, referer=None, timeout=60):
    headers = {}
    if referer:
        headers['Referer'] = referer
    resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.content, resp.headers.get("content-type", "")

def parse_pdf_and_compute_sum(pdf_bytes, column_name="value", page_number=2):
    """
    Open pdf bytes with pdfplumber, find tables on the requested page,
    try to find a column matching column_name (case-insensitive) and sum it.
    page_number is 1-indexed.
    """
    s = io.BytesIO(pdf_bytes)
    with pdfplumber.open(s) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            raise ValueError("Invalid page_number")
        page = pdf.pages[page_number - 1]
        tables = page.extract_tables()
        # convert tables to DataFrames and try to find column
        for table in tables:
            df = pd.DataFrame(table[1:], columns=table[0])
            for col in df.columns:
                if col and column_name.lower() in col.lower():
                    # coerce to numeric
                    series = pd.to_numeric(df[col].str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')
                    return float(series.sum(skipna=True))
    raise ValueError("Could not find column to sum in PDF")

def solve_quiz_page(page_content, page_url, session):
    """
    Try to infer the task and compute an answer object to submit.
    This function contains heuristics for many typical tasks: 
     - base64 encoded JSON in <pre>
     - PDF download + sum of column
     - simple extraction of text / boolean
    Returns a dict with at minimum 'submit_url' and 'answer' keys if possible.
    """
    info = find_submit_info(page_content, page_url)
    # Case: raw JSON embedded
    if "raw" in info:
        data = info["raw"]
        # Expect keys: email, secret, url, answer instructions
        # If an "answer" key already present, maybe they want it posted back.
        # We'll attempt a generic flow: if "url" present and they ask for sum, do it.
        if isinstance(data, dict) and data.get("url"):
            # If the JSON includes "answer" instructions as text, try to parse
            if "answer" in data and isinstance(data["answer"], (int, float, str, bool, dict)):
                # It's already answered (unlikely). Return as-is.
                return {"submit_url": data.get("submit", data.get("submit_url", None) or "https://example.com/submit"),
                        "answer": data["answer"]}
            # If the JSON instructs to compute the sum of 'value' column in page 2, parse that text
            # Try to match patterns in the page_content (the human instructions)
    # Case: if code word provided inline
    if "code_word" in info:
        return {"submit_url": page_url, "answer": info["code_word"]}

    # As fallback: attempt to find links to downloadable files (pdf) and a submit link
    pdf_links = re.findall(r'href=["\']([^"\']+\.pdf)["\']', page_content, re.I)
    submit_links = re.findall(r'href=["\']([^"\']*submit[^"\']*)["\']', page_content, re.I)
    # canonicalize links
    pdf_links = [urljoin(page_url, p) for p in pdf_links]
    submit_links = [urljoin(page_url, s) for s in submit_links]

    # If we see a text asking "What is the sum of the 'value' column in the table on page 2?"
    if re.search(r"sum of the .*value.* column.*page\s*2", page_content, re.I) and pdf_links:
        # download first pdf and compute
        pdf_url = pdf_links[0]
        logging.info("Detected PDF sum question. Downloading %s", pdf_url)
        content, ct = download_file(session, pdf_url, referer=page_url)
        try:
            total = parse_pdf_and_compute_sum(content, column_name="value", page_number=2)
            submit_url = submit_links[0] if submit_links else None
            return {"submit_url": submit_url, "answer": total}
        except Exception as e:
            logging.exception("PDF parsing failed: %s", e)

    # Last fallback: return minimal info so caller can make a decision
    return {"submit_url": submit_links[0] if submit_links else None, "html": page_content}

@app.route("/api/quiz", methods=["POST"])
def quiz_webhook():
    # Check JSON validity
    data = json_or_400(request)
    logging.info("Received payload: %s", {k: v for k, v in data.items() if k != "url"})

    # Validate required fields
    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")
    if not (email and secret and url):
        abort(400, description="Missing email, secret, or url")

    # Verify secret
    if secret != EXPECTED_SECRET:
        return jsonify({"ok": False, "error": "invalid secret"}), 403

    # At this point, respond 200 quickly to acknowledge receipt per spec, then process.
    # But spec requires the endpoint to perform the solving and submit the answer within 3 minutes.
    # We'll attempt to perform synchronous processing now and then return a 200 if the secret matched.
    try:
        # Use a requests session for downloads and submit posts
        session = requests.Session()
        # Use Playwright to load and render the JS page
        logging.info("Launching headless browser to render page: %s", url)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=BROWSER_TIMEOUT*1000)
            except PlaywrightTimeoutError:
                # try a load without strict wait
                page.goto(url, timeout=BROWSER_TIMEOUT*1000)
            # Get rendered content
            content = page.content()
            rendered_url = page.url
            logging.info("Page rendered. URL: %s", rendered_url)
            # Solve heuristics:
            solve_result = solve_quiz_page(content, rendered_url, session)
            logging.info("Solve result heuristics: %s", solve_result.keys())
            submit_url = solve_result.get("submit_url") or data.get("submit") or data.get("submit_url")
            answer = solve_result.get("answer")
            # If no answer determined, try extracting a JSON payload from page that instructs the submit
            if not submit_url:
                # try to find it in any links
                found = re.search(r"https?://[^\s'\"<>]+/submit[^\s'\"<>]*", content)
                if found:
                    submit_url = found.group(0)

            if not submit_url:
                logging.warning("No submit URL found. Aborting solving.")
                # Return 200 acknowledging receipt (spec requires 200 if secret matches)
                return jsonify({"ok": True, "message": "Secret verified; no submit URL discovered."}), 200

            # Build the payload to post. Basic structure required by spec:
            payload = {
                "email": email,
                "secret": secret,
                "url": url
            }
            if answer is not None:
                payload["answer"] = answer
            else:
                # Fallback: attempt to extract an answer string from instructions (e.g., code_word)
                # We'll try to look for a direct phrase pattern in the page.
                m = re.search(r'Answer[:\s]+([A-Za-z0-9\-\_ ]+)', content, re.I)
                if m:
                    payload["answer"] = m.group(1).strip()
                else:
                    payload["answer"] = ""  # empty answer if nothing else

            logging.info("Submitting answer to %s payload keys: %s", submit_url, list(payload.keys()))
            resp = session.post(submit_url, json=payload, timeout=40)
            try:
                resp.raise_for_status()
            except Exception as e:
                logging.exception("Error posting answer: %s", e)
                return jsonify({"ok": True, "message": "Secret verified but submit POST failed", "submit_status": resp.status_code}), 200

            # If submit returns JSON, forward it
            try:
                j = resp.json()
            except Exception:
                j = {"status_text": resp.text, "status_code": resp.status_code}
            logging.info("Submission response: %s", j)
            # Return 200 acknowledging secret correct (spec requires 200)
            return jsonify({"ok": True, "submit_response": j}), 200
    except Exception as e:
        logging.exception("Processing failed: %s", e)
        return jsonify({"ok": True, "message": "Secret verified but processing failed", "error": str(e)}), 200

if __name__ == "__main__":
    # For production, run under gunicorn with HTTPS (Cloud Run, Render, etc.)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
