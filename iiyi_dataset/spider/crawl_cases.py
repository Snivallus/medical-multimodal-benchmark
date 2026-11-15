"""Crawl full case content and images from iiyi.com.

The site serves the same cases through two templates:

* desktop: ``https://bingli.iiyi.com/show/{case_id}-1.html``
  A newer template whose anchors are ``.case_details`` / ``.case_summary`` /
  ``.case_study``. It usually exposes the largest image set because the
  gallery (``.case_focus_map``) is rendered on top of the inline images.
* mobile:  ``https://m.iiyi.com/bl/d-{case_id}.html``
  The legacy template whose anchors are ``.article-details`` / ``.abstract`` /
  ``.case-container``.

Both templates are supported. ``parse_case`` reads the template implied by the
input URL and falls back to the other one when a page cannot be parsed, so a
crawl survives a redesign of either side.

Every case is written with the same schema as before::

    case_idx, url, images, 标题, 关键词, 简要描述, 摘要, 内容
"""

import argparse
import json
import os
import random
import re
import shutil
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

# Request headers that mimic a regular browser, so the site does not reject us.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

DESKTOP_CASE_URL = "https://bingli.iiyi.com/show/{case_id}-1.html"
MOBILE_CASE_URL = "https://m.iiyi.com/bl/d-{case_id}.html"

# Images below this size are UI icons or tracking pixels, not case media.
MIN_IMAGE_BYTES = 5 * 1024

# Thumbnail suffixes used by the image CDN; ``abc_w150.jpg`` is a preview of
# ``abc.jpg``, so both URLs must collapse to the same deduplication key.
THUMBNAIL_SUFFIXES = ("_w150", "_w300", "_w600")

ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")

# The site answers a request burst with a verification wall instead of the case.
VERIFICATION_MARKER = "请先完成图片验证"

# How often a verification wall is retried before the next template is tried.
# The wall is sticky once the site starts rate-limiting, so one cheap retry is
# enough; anything longer only delays the fallback that usually succeeds.
VERIFICATION_RETRIES = 1

# A single connection pool keeps a multi-hour crawl from reopening sockets.
_session = requests.Session()
_session.headers.update(HEADERS)

# Counts verification walls seen so far; the crawl driver throttles harder once
# the site starts rate-limiting, which keeps a long run from being blocked for
# good.
_verification_walls = 0


# ======================
# HTTP helpers
# ======================
def fetch_soup(url, referer=None, timeout=(10, 30)):
    """Download ``url`` and return a parsed document.

    The site sometimes omits the charset in its response headers; in that case
    ``requests`` falls back to ISO-8859-1 and every Chinese character turns into
    mojibake, so the detected encoding is applied explicitly.
    """

    headers = {"Referer": referer} if referer else None
    response = _session.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"

    return BeautifulSoup(response.text, "html.parser")


def is_verification_page(soup):
    """Return ``True`` when the site served a verification wall, not a case."""

    return VERIFICATION_MARKER in soup.get_text(" ", strip=True)


def fetch_case_page(url):
    """Fetch a case page, waiting out the anti-bot verification wall.

    A burst of requests makes the site answer with a small verification page
    instead of the case. One paused retry absorbs a transient wall; if it
    survives, the caller falls back to the other template.
    """

    global _verification_walls

    soup = fetch_soup(url)

    for attempt in range(VERIFICATION_RETRIES):

        if not is_verification_page(soup):
            return soup

        _verification_walls += 1

        wait = random.uniform(5, 10) * (attempt + 1)
        print(f"verification wall on {url}, waiting {wait:.0f}s")
        time.sleep(wait)

        soup = fetch_soup(url)

    if is_verification_page(soup):
        _verification_walls += 1

    return soup


def download_image(url, save_path, referer=None, timeout=(10, 30)):
    """Download one image and save it to ``save_path``.

    Returns ``True`` only when a non-empty body was written.
    """

    headers = {"Referer": referer} if referer else None

    try:
        response = _session.get(url, headers=headers, timeout=timeout)

        # Only a complete 200 response is worth saving.
        if response.status_code != 200 or not response.content:
            return False

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True

    except requests.RequestException as error:
        print(f"image download failed: {url} ({error})")
        return False


# ======================
# Text helpers
# ======================
def normalize_block(text):
    """Normalize one block of case text.

    Non-breaking spaces are turned into plain spaces, runs of spaces are
    collapsed, and blank lines are dropped, while intentional line breaks are
    kept because the site uses them to separate paragraphs inside one section.
    """

    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    return "\n".join(line for line in lines if line)


def strip_brackets(text):
    """Remove the decorative ``【】`` brackets the site wraps headings in."""

    return re.sub(r"[【】]", "", text).strip()


def section_text_after(container, heading):
    """Collect the text of ``container`` that follows ``heading``.

    The desktop template keeps a section's value as bare text nodes after the
    ``<h3>`` heading, optionally split by ``<p>`` and ``<br>`` elements, so the
    heading itself has to be skipped explicitly.
    """

    blocks = []
    seen_heading = False

    for node in container.children:

        if node is heading:
            seen_heading = True
            continue

        # Text that appears before the heading does not belong to the section.
        if not seen_heading:
            continue

        if isinstance(node, NavigableString):
            text = normalize_block(str(node))
            if text:
                blocks.append(text)
            continue

        name = getattr(node, "name", None)

        if name == "br":
            continue

        if name in ("p", "div", "span", "li"):
            text = normalize_block(node.get_text(" ", strip=True))
            if text:
                blocks.append(text)

    return "\n".join(blocks)


# ======================
# Image helpers
# ======================
def is_case_image(url):
    """Return ``True`` when ``url`` belongs to a case's media library.

    Case images are served from ``.../bingli/...``. Everything else on the page
    is site chrome: share icons, the doctor avatar, the QR code and banners.
    """

    if not url or url.startswith(("data:", "javascript:")):
        return False

    return "/bingli/" in url


def is_thumbnail(url):
    """Return ``True`` when ``url`` points at a downscaled preview."""

    stem = os.path.splitext(url.split("?")[0])[0]

    return any(stem.endswith(suffix) for suffix in THUMBNAIL_SUFFIXES)


def image_dedupe_key(url):
    """Collapse a thumbnail URL onto the key of its full-size original."""

    key = url.split("?")[0]
    stem, extension = os.path.splitext(key)

    for suffix in THUMBNAIL_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + extension

    return key


def image_download_candidates(url):
    """Return the URLs to try for one image, best first.

    Some pages only reference the ``_w150`` preview while the full-size file is
    still online, so the original is tried before the preview. Preview files
    are usually below ``MIN_IMAGE_BYTES`` and would be dropped.
    """

    if not is_thumbnail(url):
        return [url]

    stem, extension = os.path.splitext(url.split("?")[0])

    for suffix in THUMBNAIL_SUFFIXES:
        if stem.endswith(suffix):
            return [stem[: -len(suffix)] + extension, url]

    return [url]


def collect_image_urls(soup, site, page_url):
    """Return the de-duplicated image URLs of one case page.

    On the desktop template the gallery repeats every image (large preview plus
    thumbnails) and the inline images duplicate part of the gallery, so the
    list is de-duplicated by :func:`image_dedupe_key`.
    """

    if site == "mobile":
        selectors = [".case-container img"]
    else:
        # Gallery first: it is the most complete view of the case media set.
        selectors = [".case_focus_map img", ".case_study img"]

    picked = {}

    for selector in selectors:
        for img in soup.select(selector):
            src = img.get("src") or img.get("data-src")
            if not src:
                continue

            url = urljoin(page_url, src)
            if not is_case_image(url):
                continue

            picked.setdefault(image_dedupe_key(url), url)

    return list(picked.values())


def image_extension(url):
    """Pick a file extension for a downloaded image, defaulting to ``.jpg``."""

    extension = os.path.splitext(urlparse(url).path)[1].lower()

    return extension if extension in ALLOWED_IMAGE_EXTENSIONS else ".jpg"


def looks_like_image(path):
    """Check the file signature, not just its size.

    A missing file can still answer ``200`` with an HTML error page, and saving
    that as ``img_00.jpg`` would poison the dataset.
    """

    with open(path, "rb") as f:
        head = f.read(12)

    if head.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"BM")):
        return True

    # WebP is a RIFF container.
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP"


def save_case_images(image_urls, page_url, case_idx, image_root):
    """Download the images of one case into ``image_root/case_XXXX``.

    Returns the list of saved relative paths. The case folder is removed again
    when nothing could be saved, so "has images" always means "has files".
    """

    if not image_urls:
        return []

    folder_name = f"case_{case_idx:04d}"
    case_img_dir = os.path.join(image_root, folder_name)
    os.makedirs(case_img_dir, exist_ok=True)

    saved = []

    for i, img_url in enumerate(image_urls):

        file_name = f"img_{i:02d}{image_extension(img_url)}"
        img_path = os.path.join(case_img_dir, file_name)

        # Try the full-size file first, then the preview the page referenced.
        for candidate in image_download_candidates(img_url):

            if not download_image(candidate, img_path, referer=page_url):
                continue

            # Drop icons and error pages that slipped through the URL filter.
            if os.path.getsize(img_path) < MIN_IMAGE_BYTES or not looks_like_image(img_path):
                os.remove(img_path)
                continue

            saved.append(os.path.join(folder_name, file_name).replace("\\", "/"))
            break

    if not saved:
        shutil.rmtree(case_img_dir, ignore_errors=True)

    return saved


# ======================
# Template parsers
# ======================
def parse_h5_sections(h5s):
    """Read the ``<h5>`` sub-headings of one mobile section.

    A sub-heading owns every following ``<p>`` until the next sub-heading. The
    old parser kept only the first paragraph, which silently truncated long
    fields that the site splits into several paragraphs, e.g. the physical
    examination.
    """

    sections = {}

    for h5 in h5s:

        sub_name = strip_brackets(h5.get_text(strip=True))
        values = []

        for sibling in h5.find_next_siblings():

            # The next sub-heading starts the next field.
            if sibling.name in ("h5", "div"):
                break

            if sibling.name != "p":
                continue

            text = normalize_block(sibling.get_text(" ", strip=True))
            if text:
                values.append(text)

        if values:
            sections[sub_name] = "\n".join(values)

    return sections


def parse_mobile_case(soup, url, case_idx):
    """Parse the legacy mobile template."""

    case = {"case_idx": case_idx, "url": url}

    # 1. Title
    title_tag = soup.select_one(".article-details h1")
    case["标题"] = title_tag.get_text(strip=True) if title_tag else ""

    # 2. Keywords and description
    kw = soup.find("meta", {"name": "keywords"})
    desc = soup.find("meta", {"name": "description"})

    case["关键词"] = kw["content"] if kw and kw.get("content") else ""
    case["简要描述"] = desc["content"] if desc and desc.get("content") else ""

    # 3. Summary: <span>【key】</span><var>value</var>
    abstract = {}

    for p in soup.select(".abstract p"):

        key_tag = p.find("span")
        val_tag = p.find("var")

        if key_tag and val_tag:
            key = strip_brackets(key_tag.get_text(strip=True))
            abstract[key] = normalize_block(val_tag.get_text(strip=True))

    case["摘要"] = abstract

    # 4. Case body: <h3>【group】</h3> followed by one <div> per section
    content = {}
    container = soup.select_one(".case-container")

    if container:

        for sec in container.find_all("h3"):

            sec_name = strip_brackets(sec.get_text(strip=True))
            sec_content = {}
            text_buffer = []

            for node in sec.find_next_siblings():

                # The next group heading ends this section.
                if node.name == "h3":
                    break

                # The copyright block ends the case body.
                if node.get("class") and "Copyright-notice" in node.get("class"):
                    break

                if node.name != "div":
                    continue

                # Some sections use <h5> sub-headings and become dictionaries.
                h5s = node.find_all("h5")

                if h5s:
                    sec_content.update(parse_h5_sections(h5s))

                # Otherwise the whole div is plain text.
                else:
                    text = normalize_block(node.get_text(" ", strip=True))
                    if text:
                        text_buffer.append(text)

            if sec_content:
                content[sec_name] = sec_content
            elif text_buffer:
                content[sec_name] = "\n".join(text_buffer)

    case["内容"] = content
    case["images"] = []

    return case


def parse_desktop_case(soup, url, case_idx):
    """Parse the newer desktop template."""

    case = {"case_idx": case_idx, "url": url}

    # 1. Title
    title_tag = soup.select_one(".case_details_cont > h2") or soup.select_one(".case_details_cont h2")
    case["标题"] = title_tag.get_text(strip=True) if title_tag else ""

    # 2. Keywords and description
    kw = soup.find("meta", {"name": "keywords"})
    desc = soup.find("meta", {"name": "description"})

    case["关键词"] = kw["content"] if kw and kw.get("content") else ""
    case["简要描述"] = desc["content"] if desc and desc.get("content") else ""

    # 3. Summary: <var>【key】</var><span>value</span>
    abstract = {}

    for p in soup.select(".case_summary .situation p"):

        key_tag = p.find("var")
        val_tag = p.find("span")

        if key_tag and val_tag:
            key = strip_brackets(key_tag.get_text(strip=True))
            abstract[key] = normalize_block(val_tag.get_text(" ", strip=True))

    case["摘要"] = abstract

    # 4. Case body: <div class="case_study"><h2>【group】</h2><div><h3>section</h3>value</div>...
    content = {}

    for group in soup.select(".case_study"):

        head = group.find("h2")
        group_name = strip_brackets(head.get_text(strip=True)) if head else ""

        if not group_name:
            continue

        sections = {}
        loose_text = []

        for node in group.children:

            if isinstance(node, NavigableString):
                text = normalize_block(str(node))
                if text:
                    loose_text.append(text)
                continue

            if getattr(node, "name", None) != "div":
                continue

            h3 = node.find("h3")

            # A div with an <h3> is a named section.
            if h3:
                sec_name = strip_brackets(h3.get_text(strip=True))
                sections[sec_name] = section_text_after(node, h3)

            # A div without an <h3> is free text, e.g. the analysis/summary group.
            else:
                text = normalize_block(node.get_text(" ", strip=True))
                if text:
                    loose_text.append(text)

        if sections:
            content[group_name] = sections
        elif loose_text:
            content[group_name] = "\n".join(loose_text)

    case["内容"] = content
    case["images"] = []

    return case


TEMPLATE_PARSERS = {
    "mobile": parse_mobile_case,
    "desktop": parse_desktop_case,
}


# ======================
# URL / template dispatch
# ======================
def case_id_from_url(url):
    """Extract the numeric case id from either template's URL."""

    match = re.search(r"/show/(\d+)-", url) or re.search(r"/d-(\d+)\.html", url)

    return match.group(1) if match else None


def candidate_pages(url, prefer="auto"):
    """Return ``(url, site)`` pairs to try for this case, best first.

    ``prefer`` forces a template; otherwise the template of the input URL is
    tried first and the other one is used as a fallback.
    """

    case_id = case_id_from_url(url)

    if case_id:
        urls = {
            "mobile": MOBILE_CASE_URL.format(case_id=case_id),
            "desktop": DESKTOP_CASE_URL.format(case_id=case_id),
        }
    else:
        # Unknown URL shape: only the URL we were given can be used.
        urls = {"mobile" if "m.iiyi.com" in url else "desktop": url}

    if prefer in ("mobile", "desktop"):
        site = prefer
    else:
        site = "mobile" if "m.iiyi.com" in url else "desktop"

    order = ["mobile", "desktop"] if site == "mobile" else ["desktop", "mobile"]

    return [(urls[name], name) for name in order if urls.get(name)]


def parse_case(url, case_idx, image_root, prefer="auto"):
    """Parse one case page and download its images.

    Returns ``(case, has_img)``. ``has_img`` is ``True`` only when at least one
    image file was actually saved.
    """

    errors = []
    pages = candidate_pages(url, prefer)
    primary_site = pages[0][1]

    for page_url, site in pages:

        try:
            soup = fetch_case_page(page_url)
        except requests.RequestException as error:
            errors.append(f"{page_url}: {error}")
            continue

        case = TEMPLATE_PARSERS[site](soup, page_url, case_idx)

        # A usable page always yields a title and a body.
        if not case.get("标题") or not (case.get("内容") or case.get("摘要")):
            reason = "verification wall" if is_verification_page(soup) else "empty case body"
            errors.append(f"{page_url}: {reason}")
            continue

        # Keep the URL the caller asked for, not the fallback that worked.
        case["url"] = url

        image_urls = collect_image_urls(soup, site, page_url)
        saved = save_case_images(image_urls, page_url, case_idx, image_root)
        case["images"] = saved

        if site != primary_site:
            print(f"parsed via fallback template ({site}): {page_url}")

        return case, bool(saved)

    raise RuntimeError("; ".join(errors) or f"no candidate page for {url}")


# ======================
# Incremental storage
# ======================
def load_records(file_path):
    """Load records from a JSON array or from a JSONL checkpoint.

    A crawl that gets killed mid-write leaves a truncated checkpoint; anything
    that cannot be parsed is dropped so the rerun simply starts from the last
    complete line.
    """

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []

    if not text.strip():
        return []

    # The final output is a JSON object with a "cases" key.
    if text.lstrip().startswith("{"):

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict):
            return data.get("cases") or []

        return []

    # A JSON array (the old checkpoint format).
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        return data

    # Anything else is JSONL.
    records = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records


def append_record(file_path, data):
    """Append one record to a JSONL checkpoint.

    Writing a single line keeps the cost constant per case. Rewriting a growing
    JSON array, as the old version did, turns a few thousand cases into gigabytes
    of redundant writes.
    """

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()


# ======================
# Crawl driver
# ======================
def scrape_cases(
    input_json,
    output_json,
    image_root,
    failed_json,
    temp_json,
    link_field="url",
    prefer="auto",
    limit=None,
    resume=True,
):
    """Crawl every case listed in ``input_json``.

    Parameters
    ----------
    input_json : str
        List of cases with a URL. Each entry needs ``link_field`` or ``url``.
    output_json : str
        Final output, ``{"metadata": ..., "cases": [...]}``.
    image_root : str
        Root folder for the per-case image folders.
    failed_json : str
        Cases that could not be crawled.
    temp_json : str
        Accumulator that makes a killed crawl resumable.
    link_field : str
        Which field of the input entries holds the case URL. ``url`` is the
        desktop template and is the default because it exposes the whole image
        gallery; ``alternate_url`` selects the mobile template.
    prefer : str
        ``auto``, ``mobile`` or ``desktop``. ``auto`` follows the input URL and
        falls back to the other template.
    limit : int or None
        Stop after this many new cases; useful for smoke tests.
    resume : bool
        Skip cases already present in ``temp_json``.
    """

    with open(input_json, encoding="utf-8") as f:
        cases = json.load(f)

    # The temp file doubles as the resume checkpoint. It is removed once the
    # output is written, so a finished run is resumed from the output file
    # instead of crawling every case again.
    if resume:
        results = load_records(temp_json) or load_records(output_json)

        if results and not os.path.exists(temp_json):
            for record in results:
                append_record(temp_json, record)
    else:
        results = []
        if os.path.exists(temp_json):
            os.remove(temp_json)

    done_urls = {case.get("url") for case in results if case.get("url")}

    os.makedirs(image_root, exist_ok=True)

    started = len(results)
    failed_before = len(load_records(failed_json))
    skipped = 0
    failed = 0
    processed = 0

    for idx, case in enumerate(cases):

        url = case.get(link_field) or case.get("url")

        if not url:
            continue

        if resume and url in done_urls:
            skipped += 1
            continue

        if limit is not None and processed >= limit:
            break

        print(f"\nprocessing case {idx}: {url}")

        success = False
        walls_before = _verification_walls

        # Three attempts absorb transient network and server errors.
        for attempt in range(3):

            try:
                data, _ = parse_case(url, idx, image_root, prefer=prefer)

                append_record(temp_json, data)
                results.append(data)
                done_urls.add(url)

                processed += 1
                success = True
                break

            except Exception as error:
                print(f"attempt {attempt + 1} failed: {error}")
                time.sleep(random.uniform(3, 5))

        if not success:
            failed += 1
            append_record(failed_json, case)

        # A verification wall means the site is rate-limiting this client; back
        # off hard so the rest of the crawl does not hit it case after case.
        if _verification_walls > walls_before:
            cooldown = random.uniform(30, 60)
            print(f"verification wall hit, cooling down {cooldown:.0f}s")
            time.sleep(cooldown)

        # Throttle: the site bans clients that hammer it.
        sleep_time = random.uniform(1, 2)
        print(f"sleep {sleep_time:.2f}s")
        time.sleep(sleep_time)

        # Occasionally pause much longer to look like a human reader.
        if random.random() < 0.1:
            long_sleep = random.uniform(5, 10)
            print(f"long sleep {long_sleep:.2f}s")
            time.sleep(long_sleep)

    cases_with_images = sum(1 for case in results if case.get("images"))

    metadata = {
        "total_cases": len(results),
        "cases_with_images": cases_with_images,
        "cases_without_images": len(results) - cases_with_images,
        "failed_cases": failed_before + failed,
        "new_cases_this_run": len(results) - started,
        "skipped_existing_cases": skipped,
    }

    output_data = {
        "metadata": metadata,
        "cases": results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # The checkpoint is no longer needed once the final file is written.
    if os.path.exists(temp_json):
        os.remove(temp_json)

    print("\nDone")
    print("total:", metadata["total_cases"])
    print("with images:", metadata["cases_with_images"])
    print("failed:", failed)


def main():
    parser = argparse.ArgumentParser(description="Crawl iiyi case pages and images.")

    parser.add_argument("--input", default="merged_cases.json", help="case list to crawl")
    parser.add_argument("--output", default="succeeded_cases.json", help="final output file")
    parser.add_argument("--images", default="raw_images", help="root folder for images")
    parser.add_argument("--failed", default="failed_cases.json", help="cases that failed")
    parser.add_argument("--temp", default="temp.jsonl", help="resume checkpoint (JSONL)")
    parser.add_argument(
        "--link-field",
        default="url",
        choices=["url", "alternate_url"],
        help="input field holding the case URL; url is the desktop template, "
        "alternate_url the mobile one",
    )
    parser.add_argument(
        "--site",
        default="auto",
        choices=["auto", "mobile", "desktop"],
        help="template to read; auto follows the input URL and falls back",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N new cases")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore the checkpoint and crawl every input case again",
    )

    args = parser.parse_args()

    scrape_cases(
        input_json=args.input,
        output_json=args.output,
        image_root=args.images,
        failed_json=args.failed,
        temp_json=args.temp,
        link_field=args.link_field,
        prefer=args.site,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
