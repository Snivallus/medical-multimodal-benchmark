"""Classify every shipped image so a human knows which items are worth reading.

The photo condition is only meaningful when the image carries information the
text does not. The 2024 crawl shipped whatever the page showed, which turns out
to include laboratory printouts and intra-operative photos with the conclusion
burned in as a caption, so the photo score partly measures OCR rather than
visual reasoning.

This script asks the vision model one narrow question per image — what kind of
picture is this, what does it show, and transcribe the text — without showing it
the gold diagnosis, so the description cannot be nudged toward the answer. The
answer key is then matched against the transcript locally, which keeps the leak
test deterministic and auditable.

    python bench/triage_images.py --workers 6
    python bench/triage_images.py --limit 6          # smoke test
    python bench/triage_images.py --report-only      # rebuild the markdown

Outputs ``bench/results/image_triage.jsonl`` (raw, git-ignored) and
``bench/IMAGE_TRIAGE.md`` (the human shortlist, tracked).
"""

import argparse
import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import llm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "bench" / "data"
ITEMS = DATA / "items.jsonl"
IMAGE_ROOT = DATA / "images"
RAW = REPO_ROOT / "bench" / "results" / "image_triage.jsonl"
REPORT = REPO_ROOT / "bench" / "IMAGE_TRIAGE.md"

VISION_MODEL = "Qwen3.8-Max"

# One call per image with a long read timeout: a 1.8 MB photo takes a while to
# upload, and the reasoning models at this endpoint routinely need 40-50 s.
TIMEOUT = (15, 240)

MODALITY_PROMPT = """你是一名影像科/病理科医师。请只描述这张图片本身，不要推测临床诊断。

只输出一个 JSON 对象，字段如下：

- "modality": 从 ["影像", "病理切片", "内镜", "检验报告", "病历文本截图", "临床照片", "示意/其他"] 中选一个。
  "影像" 指 X线、CT、MRI、超声、钼靶、造影等；"检验报告" 指化验单、检查申请单等以数值或表格为主的打印件；
  "病历文本截图" 指以文字为主、几乎没有医学图像的截图；"临床照片" 指体表、术中、术后等真实拍摄的照片。
- "findings": 一两句描述图上可见的阳性征象（部位、大小、形态、信号、颜色等）。若看不清或无关，写 "无法判读"。
- "text": 逐字转录图中所有可读文字（标题、标签、箭头旁说明、备注，含英文）。没有文字填空字符串。最多 200 字。
- "text_kind": 从 ["结论", "标注", "无"] 中选一个。
  "结论" 指文字里直接写出了诊断、检查结论或手术方式（例如 "考虑…"、"符合…"、"诊断：…"、"…术后"）；
  "标注" 指只有姓名、医院、日期、部位名称、编号、箭头说明等；
  "无" 指没有可读文字。
"""

SYSTEM_PROMPT = "你是严谨的医学图像标注员，只描述看到的内容，不做诊断推断，也不臆造文字。"

# A label is leaked when the transcript repeats the diagnosis. Chinese has no
# whitespace, so the match is done on the raw characters; medical terms often
# carry bracketed qualifiers that the caption would not reproduce, hence the
# fallback to a distinctive prefix.
MIN_FRAGMENT = 4

_lock = threading.Lock()


def load_items():
    """Return the items, keyed by the image paths they reference."""

    items = []

    with open(ITEMS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    return items


def image_mime(data):
    """Guess the type from the signature; the crawl names every file .jpg."""

    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"GIF8"):
        return "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"BM"):
        return "bmp"

    return "jpeg"


def ask(prompt, raw, max_tokens=700, label=""):
    """Send one image and one prompt, and return the parsed JSON answer.

    ``raw`` is passed as bytes so a caller can ask about an image it has just
    cropped in memory, without writing a temporary file first.
    """

    data = base64.b64encode(raw).decode("ascii")
    mime = image_mime(raw[:16])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}},
            ],
        },
    ]

    payload = {
        "model": VISION_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {llm.api_key()}",
        "Content-Type": "application/json",
    }

    url = f"{llm.api_root()}/chat/completions"
    last_error = None

    # No shared session here: the batch runs on threads and requests.Session is
    # not thread-safe, so each call opens its own connection.
    for attempt in range(llm.DEFAULT_RETRIES):

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)

            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"]
                parsed = llm.extract_json(text)

                if parsed is None:
                    raise ValueError(f"no JSON in answer: {text[:200]}")

                return parsed

            if response.status_code not in llm.RETRYABLE_STATUS:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

            last_error = f"HTTP {response.status_code}"

        except (requests.RequestException, ValueError, KeyError, IndexError) as error:
            last_error = f"{type(error).__name__}: {error}"

        if attempt < llm.DEFAULT_RETRIES - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"{label}: gave up after {llm.DEFAULT_RETRIES} attempts ({last_error})")


def describe(path):
    """Ask the vision model to describe one image file."""

    return ask(MODALITY_PROMPT, Path(path).read_bytes(), label=Path(path).name)


def normalize(text):
    """Fold the text for matching: full-width punctuation and spaces removed."""

    text = (text or "").lower()
    text = re.sub(r"[\s，。、；：,.;:!?！？（）()\[\]【】\-—_/]", "", text)

    return text


def fragments(diagnosis):
    """Return the strings worth searching for, longest first."""

    clean = normalize(diagnosis)
    parts = [clean] if len(clean) >= MIN_FRAGMENT else []

    # "左膝前交叉韧带断裂" also appears as "前交叉韧带断裂", and captions often
    # drop the side, so a trimmed variant is worth checking too.
    if len(clean) > 6:
        parts.append(clean[1:])
        parts.append(clean[:-2])

    return [part for part in dict.fromkeys(parts) if len(part) >= MIN_FRAGMENT]


def leaked_labels(record, item):
    """Return the gold labels whose characters appear in the image text."""

    transcript = normalize(record.get("text", ""))

    if not transcript:
        return []

    hits = []

    for label in [item.get("main_diagnosis", "")] + item.get("diagnoses", []):
        if not label:
            continue

        for fragment in fragments(label):
            if fragment in transcript:
                hits.append(label)
                break

    return list(dict.fromkeys(hits))


def tier(record, leaks):
    """Bucket one image by how usable it is as visual evidence."""

    modality = record.get("modality", "")
    text_kind = record.get("text_kind", "")
    findings = (record.get("findings") or "").strip()

    if modality in ("检验报告", "病历文本截图"):
        return "C"

    if text_kind == "结论" or leaks:
        return "C"

    if modality == "示意/其他" or findings in ("", "无法判读"):
        return "B"

    if modality in ("影像", "病理切片", "内镜") and text_kind in ("标注", "无"):
        return "A"

    return "B"


def load_raw(keep_errors=False):
    """Read the records already done, so an interrupted run can resume.

    With ``keep_errors=False`` a record that failed is treated as not done, which
    is what ``--retry-errors`` needs: the timeout it hit is often transient.
    """

    records = {}

    if not RAW.exists():
        return records

    with open(RAW, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("error") and not keep_errors:
                continue

            records[record["image"]] = record

    return records


def run(workers, limit, retry_errors=False):
    """Describe every image not already in the raw file."""

    RAW.parent.mkdir(parents=True, exist_ok=True)

    items = load_items()

    # A failed record normally still counts as done so a rerun does not hammer
    # the same broken call; --retry-errors is the deliberate exception.
    done = load_raw(keep_errors=not retry_errors)

    todo = []

    for item in items:
        for relative in item["images"]:
            if relative not in done:
                todo.append((item["item_id"], relative))

    # An image shared by two items would otherwise be fetched twice.
    todo = list(dict.fromkeys(todo))

    if limit:
        todo = todo[:limit]

    print(f"{len(done)} image(s) already done, {len(todo)} to go", flush=True)

    if not todo:
        return

    started = time.time()
    counter = {"n": 0}
    total = len(todo)
    file_lock = threading.Lock()

    def work(job):
        item_id, relative = job

        try:
            record = describe(IMAGE_ROOT / relative)
            error = None
        except Exception as exception:  # noqa: BLE001 - keep the batch alive
            record, error = {}, f"{type(exception).__name__}: {exception}"

        record = {
            "image": relative,
            "item_id": item_id,
            "modality": record.get("modality", ""),
            "findings": record.get("findings", ""),
            "text": record.get("text", ""),
            "text_kind": record.get("text_kind", ""),
            "error": error,
        }

        with file_lock:
            with open(RAW, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            counter["n"] += 1
            elapsed = time.time() - started
            rate = elapsed / counter["n"]
            print(
                f"[{counter['n']}/{total}] {relative} {record['modality']} "
                f"{record['text_kind']} ({rate:.0f}s each, ~{rate * (total - counter['n']) / 60:.0f} min left)",
                flush=True,
            )

        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))


def build_report():
    """Join the descriptions with the answer key and write the shortlist."""

    items = load_items()
    by_image = load_raw()

    cases = []

    for item in items:
        images = []

        for relative in item["images"]:
            record = by_image.get(relative)

            if record is None:
                images.append(
                    {
                        "image": relative,
                        "modality": "未运行",
                        "findings": "",
                        "text": "",
                        "text_kind": "",
                        "leaks": [],
                        "tier": "B",
                        "error": "not triaged",
                    }
                )
                continue

            leaks = leaked_labels(record, item)
            images.append(
                {
                    "image": relative,
                    "modality": record.get("modality", ""),
                    "findings": record.get("findings", ""),
                    "text": record.get("text", ""),
                    "text_kind": record.get("text_kind", ""),
                    "leaks": leaks,
                    "tier": tier(record, leaks),
                    "error": record.get("error"),
                }
            )

        tiers = [entry["tier"] for entry in images]
        verdict = "C" if "C" in tiers else ("B" if "B" in tiers else "A")

        cases.append(
            {
                "item_id": item["item_id"],
                "main_diagnosis": item.get("main_diagnosis", ""),
                "verdict": verdict,
                "images": images,
            }
        )

    counts = {
        "images": sum(len(case["images"]) for case in cases),
        "items": len(cases),
        "A": sum(1 for case in cases if case["verdict"] == "A"),
        "B": sum(1 for case in cases if case["verdict"] == "B"),
        "C": sum(1 for case in cases if case["verdict"] == "C"),
    }
    image_counts = {"A": 0, "B": 0, "C": 0}

    for case in cases:
        for entry in case["images"]:
            image_counts[entry["tier"]] += 1

    lines = [
        "# Image triage",
        "",
        "Every shipped image described by `Qwen3.8-Max` without showing it the answer key,",
        "then compared against the gold labels locally. Generated by `bench/triage_images.py`,",
        "raw records in `bench/results/image_triage.jsonl`.",
        "",
        "This is the **current** state of `bench/data/images`. The first pass, on the raw",
        "crawl, found 181 images with 83 printed figure legends and 50 affected items",
        "(45 A / 4 B / 51 C items). `bench/strip_captions.py` has since cropped the legend",
        "off 61 images and the numbers below were rebuilt after that, with each cropped",
        "image re-described; `bench/data/caption_crops.json` records every crop.",
        "",
        "Tiers, worst image in the item decides:",
        "",
        "- **A** 影像 / 病理切片 / 内镜, text only labels or absent — usable as visual evidence.",
        "- **B** 临床照片, 示意/其他, or unreadable — weakly usable, needs a human look.",
        "- **C** 检验报告, 病历文本截图, transcribed text states the conclusion, or the gold",
        "  label is printed on the image — the photo condition is really OCR of the answer.",
        "",
        f"Items: **{counts['A']} A / {counts['B']} B / {counts['C']} C** ({counts['items']} total). "
        f"Images: {image_counts['A']} A / {image_counts['B']} B / {image_counts['C']} C "
        f"({counts['images']} total).",
        "",
        "## Items to fix first (C)",
        "",
        "| item | main diagnosis | image | modality | text kind | leak | text |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def cell(text, limit=60):
        text = (text or "").replace("|", "/").replace("\n", " ").strip()
        return text[:limit] + ("…" if len(text) > limit else "")

    for case in cases:
        if case["verdict"] != "C":
            continue

        for entry in case["images"]:
            if entry["tier"] != "C":
                continue

            lines.append(
                "| {item} | {diagnosis} | {image} | {modality} | {kind} | {leak} | {text} |".format(
                    item=case["item_id"],
                    diagnosis=cell(case["main_diagnosis"], 24),
                    image=Path(entry["image"]).name,
                    modality=entry["modality"],
                    kind=entry["text_kind"],
                    leak="yes" if entry["leaks"] else "",
                    text=cell(entry["text"], 70),
                )
            )

    lines += ["", "## Items needing a look (B)", "", "| item | main diagnosis | image | modality | findings |", "| --- | --- | --- | --- | --- |"]

    for case in cases:
        if case["verdict"] != "B":
            continue

        for entry in case["images"]:
            if entry["tier"] != "B":
                continue

            lines.append(
                "| {item} | {diagnosis} | {image} | {modality} | {findings} |".format(
                    item=case["item_id"],
                    diagnosis=cell(case["main_diagnosis"], 24),
                    image=Path(entry["image"]).name,
                    modality=entry["modality"],
                    findings=cell(entry["findings"], 60),
                )
            )

    lines += ["", "## Items that survive (A)", ""]

    survivors = [case["item_id"] for case in cases if case["verdict"] == "A"]
    lines.append(", ".join(f"`{item}`" for item in survivors) if survivors else "(none)")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {REPORT.relative_to(REPO_ROOT)}: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6, help="parallel image calls")
    parser.add_argument("--limit", type=int, default=0, help="only the first N images")
    parser.add_argument("--report-only", action="store_true", help="skip the API calls")
    parser.add_argument("--retry-errors", action="store_true", help="redo the images that failed")
    args = parser.parse_args()

    if not llm.is_configured():
        raise SystemExit("no API key; put LLM_API_KEY in bench/.env")

    if not args.report_only:
        run(args.workers, args.limit, args.retry_errors)

    build_report()


if __name__ == "__main__":
    main()
