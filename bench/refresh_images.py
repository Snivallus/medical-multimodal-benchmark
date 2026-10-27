"""Re-download the case images at full size.

The 2024 crawl saved whatever file the gallery pointed at, which is the
``_w150`` thumbnail: 137 of the 181 shipped images are 200 px or smaller, while
the page's full-size file is 300-900 px. The spider already prefers the full-size
URL, so this script re-fetches the pages of the frozen dataset and overwrites the
thumbnails in place, keeping the file names ``items.jsonl`` refers to.

It only touches images that are still small unless ``--all`` is given, never
replaces a sharper file with a blurrier one, and never adds or removes images: a
page that now has more images than the item references is reported, not silently
changed.

Usage::

    python bench/refresh_images.py                 # only the thumbnail cases
    python bench/refresh_images.py --all            # every referenced case
    python bench/refresh_images.py --dry-run        # report what would happen
    python bench/refresh_images.py --limit 5        # try a handful first
"""

import argparse
import json
import os
import random
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPIDER = REPO_ROOT / "iiyi_dataset" / "spider"

# The spider is a sibling script, not an installed package, and the crawl project
# is deliberately not part of this repository: only the frozen dataset is.
sys.path.insert(0, str(SPIDER))

try:
    import crawl_cases  # noqa: E402
except ImportError as error:  # pragma: no cover - only on a clone without the crawl
    raise SystemExit(
        f"cannot import the spider from {SPIDER} ({error}).\n"
        "Re-downloading images needs the iiyi_dataset crawl project next to bench/; "
        "it is git-ignored, so a fresh clone does not have it."
    )

DATA = REPO_ROOT / "bench" / "data"

# The 2024 crawl cached 150 px thumbnails; anything at least this large is a real
# file already, and re-fetching it would at best return the same bytes.
TARGET_LONG_SIDE = 300


def image_size(path):
    """Return ``(format, width, height)`` from the file header."""

    data = Path(path).read_bytes()[:65536]

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", data[16:24])
        return "png", width, height

    if data.startswith(b"\xff\xd8"):
        index = 2
        while index < len(data) - 9:
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                height, width = struct.unpack(">HH", data[index + 5:index + 9])
                return "jpeg", width, height
            length = struct.unpack(">H", data[index + 2:index + 4])[0]
            index += 2 + length

    return "?", 0, 0


def load_jsonl(path):
    path = Path(path)

    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def referenced_cases(items, conflicts):
    """Map case folder to the image paths the dataset already references."""

    cases = {}

    for record in items + conflicts:
        for relative in record.get("images") or []:
            parts = Path(relative).parts
            if len(parts) != 2:
                continue
            folder, name = parts
            cases.setdefault(folder, {})[name] = relative

    return cases


def fetch_case_page(url, retries=3):
    """Fetch a case page, retrying transport errors with a short backoff.

    A burst of requests makes the site slow down or time out; the spider already
    waits out the verification wall, but a read timeout has to be retried here.
    """

    for attempt in range(retries):
        try:
            return crawl_cases.fetch_case_page(url)
        except Exception:  # noqa: BLE001 - retried, then reported by the caller
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1) + random.uniform(0, 1.5))

    raise RuntimeError("unreachable")


def refresh_case(folder, names, case, data_images, dry_run):
    """Re-download every image of one case, in page order."""

    url = case["url"]
    result = {"folder": folder, "refreshed": 0, "failed": 0, "kept": 0, "extra": 0, "error": None}

    try:
        soup = fetch_case_page(url)
    except Exception as exc:  # noqa: BLE001 - reported per case
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if crawl_cases.is_verification_page(soup):
        result["error"] = "verification wall"
        return result

    site = "mobile" if "m.iiyi.com" in url else "desktop"
    page_urls = crawl_cases.collect_image_urls(soup, site, url)
    result["extra"] = max(0, len(page_urls) - len(names))

    if not page_urls:
        result["error"] = "no image on page"
        return result

    for name, image_url in zip(sorted(names), page_urls):
        target = data_images / folder / name
        current = max(image_size(target)[1:]) if target.exists() else 0
        temp = target.with_name(target.name + ".new")
        status = None

        if dry_run:
            result["refreshed"] += 1
            continue

        for candidate in crawl_cases.image_download_candidates(image_url):
            if not crawl_cases.download_image(candidate, temp, referer=url):
                continue

            if temp.stat().st_size < crawl_cases.MIN_IMAGE_BYTES or not crawl_cases.looks_like_image(temp):
                temp.unlink(missing_ok=True)
                continue

            # The candidate list falls back to the thumbnail, so never replace a
            # sharper image with a blurrier one.
            if max(image_size(temp)[1:]) < current:
                temp.unlink(missing_ok=True)
                status = "kept"
                break

            os.replace(temp, target)
            status = "refreshed"
            break

        result[status or "failed"] += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Re-download the frozen dataset's images at full size.")
    parser.add_argument("--data", default=str(DATA), help="frozen dataset directory")
    parser.add_argument("--workers", type=int, default=4, help="parallel case fetches")
    parser.add_argument("--all", action="store_true", help="refresh every case, not only small images")
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases")
    parser.add_argument("--dry-run", action="store_true", help="report without downloading")
    args = parser.parse_args()

    data_dir = Path(args.data)
    data_images = data_dir / "images"

    items = load_jsonl(data_dir / "items.jsonl")
    conflicts = load_jsonl(data_dir / "conflict_items.jsonl")
    cases = {f"case_{record['case_idx']:04d}": record for record in load_jsonl(data_dir / "cases.jsonl")}

    referenced = referenced_cases(items, conflicts)
    todo = []

    for folder, names in sorted(referenced.items()):
        case = cases.get(folder)
        if case is None:
            print(f"[warn] {folder} has no case record, skipped")
            continue

        sides = [max(image_size(data_images / folder / name)[1:]) for name in sorted(names)]
        smallest = min(sides) if sides else 0

        if not args.all and smallest >= TARGET_LONG_SIDE:
            continue

        todo.append((folder, names, case, smallest))

    if args.limit:
        todo = todo[: args.limit]

    print(f"referenced cases: {len(referenced)} | to refresh: {len(todo)}")

    refreshed = failed = extra = errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(refresh_case, folder, names, case, data_images, args.dry_run): (folder, smallest)
            for folder, names, case, smallest in todo
        }

        for index, future in enumerate(as_completed(futures), 1):
            folder, smallest = futures[future]
            result = future.result()
            refreshed += result["refreshed"]
            failed += result["failed"]
            extra += 1 if result["extra"] else 0

            note = f" ({result['error']})" if result["error"] else ""
            if result["extra"]:
                note += f" page has {result['extra']} more image(s)"

            if result["error"]:
                errors += 1

            print(f"[{index:3d}/{len(todo)}] {folder} smallest={smallest}px "
                  f"refreshed={result['refreshed']} failed={result['failed']}{note}")

    print(f"\nrefreshed {refreshed} image(s), failed {failed}, cases with extra page images {extra}, errors {errors}")

    if not args.dry_run:
        print("next: python bench/freeze_data.py --manifest-only")


if __name__ == "__main__":
    main()
