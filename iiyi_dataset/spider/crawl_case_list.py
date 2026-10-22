"""Collect case titles and URLs from the iiyi case index pages.

The index pages load more cases by posting ``p=<page>`` to their own URL and
receiving a JSON payload whose ``data.html`` field holds the next block of
``<div class="li">`` entries. That endpoint replaces the browser automation the
old ``click_seemore.py`` needed (Selenium plus a matching ``msedgedriver.exe``).

Only the home feed supports paged loading; the selected and newest feeds answer the
same POST with a full HTML page, so their first page is all that is available.
"""

import argparse
import json
import os
import random
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://bingli.iiyi.com"
MOBILE_CASE_URL = "https://m.iiyi.com/bl/d-{case_id}.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# (section URL, output file) for the recommended, selected and newest feeds.
SECTIONS = [
    (f"{BASE_URL}/", "recommended_cases.json"),
    (f"{BASE_URL}/cull/", "selected_cases.json"),
    (f"{BASE_URL}/news/", "newest_cases.json"),
]

_session = requests.Session()
_session.headers.update(HEADERS)


def fetch_page(url):
    """Download one index page as HTML."""

    response = _session.get(url, timeout=(10, 30))
    response.raise_for_status()

    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"

    return response.text


def fetch_page_block(url, page):
    """Ask the index page for one more block of cases.

    Returns the HTML block, or ``None`` when this section has no paged loading.
    The endpoint answers with a JSON body but labels it ``text/html``, so the
    body is parsed instead of trusted from the Content-Type header.
    """

    response = _session.post(
        url,
        data={"p": page},
        headers={"X-Requested-With": "XMLHttpRequest"},
        timeout=(10, 30),
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        # The selected and newest feeds answer with a full HTML page instead.
        return None

    if not isinstance(payload, dict) or payload.get("code") != 200:
        return None

    return (payload.get("data") or {}).get("html") or ""


def extract_cases(html):
    """Extract ``{title, url, alternate_url}`` entries from an index block."""

    soup = BeautifulSoup(html, "html.parser")
    cases = []

    for li in soup.select(".li"):

        a_tag = li.select_one("a.name")
        title_tag = li.select_one(".sp")

        if not (a_tag and title_tag):
            continue

        href = urljoin(BASE_URL, a_tag["href"])
        match = re.search(r"/show/(\d+)-", href)
        case_id = match.group(1) if match else None

        cases.append(
            {
                "title": title_tag.get_text(strip=True),
                "url": href,
                "alternate_url": MOBILE_CASE_URL.format(case_id=case_id) if case_id else None,
            }
        )

    return cases


def crawl_section(section_url, output_json, max_pages=200):
    """Collect one section and write it to ``output_json``."""

    collected = {}

    # The first page is server-rendered.
    for case in extract_cases(fetch_page(section_url)):
        collected.setdefault(case["url"], case)

    page = 2

    while page <= max_pages:

        block = fetch_page_block(section_url, page)

        if block is None:
            print(f"[note] {section_url} has no paged loading; kept the first page only")
            break

        added = 0

        for case in extract_cases(block):
            if case["url"] not in collected:
                collected[case["url"]] = case
                added += 1

        print(f"  page {page}: +{added} new cases, total {len(collected)}")

        # An empty block, or one that only repeats what we already have, marks
        # the end of the feed.
        if added == 0:
            break

        page += 1
        time.sleep(random.uniform(0.5, 1.5))

    result = list(collected.values())

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{section_url}: {len(result)} cases -> {output_json}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Collect iiyi case index lists.")
    parser.add_argument(
        "--section",
        action="append",
        choices=[url for url, _ in SECTIONS],
        help="section to crawl; repeatable, defaults to all three",
    )
    parser.add_argument("--max-pages", type=int, default=200, help="page limit per section")
    parser.add_argument("--out-dir", default=".", help="folder for the output files")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for section_url, output_name in SECTIONS:

        if args.section and section_url not in args.section:
            continue

        print(f"\ncrawling {section_url}")

        crawl_section(
            section_url,
            os.path.join(args.out_dir, output_name),
            max_pages=args.max_pages,
        )


if __name__ == "__main__":
    main()
