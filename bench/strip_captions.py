"""Crop the printed caption off the images that leak the diagnosis.

The crawled case images are journal figure plates: the radiographs sit on top and
the article's figure legend is printed underneath, naming the diagnosis the
text-only variant was supposed to hide. Cutting the legend off restores the
situation the benchmark wants to measure - a model looking at a radiograph, not
at a caption.

The cut is not guessed from pixels. Pixel heuristics confuse a bright caption
with a bright figure, so the model is asked where the caption is and how tall it
is, and the crop is then verified with the same describe prompt
`triage_images.py` uses: the cropped image must no longer state a conclusion,
must not contain the gold label, and must still show something readable. Images
whose text is painted inside the scan (viewer overlays, annotated snapshots)
fail that test and are left as they are.

    python bench/strip_captions.py --dry-run    # ask where the captions are
    python bench/strip_captions.py --workers 4  # crop, verify, write
    python bench/freeze_data.py --manifest-only

Originals are copied to ``.tmp/originals/`` before being overwritten, and every
decision is recorded in ``bench/data/caption_crops.json``.
"""

import argparse
import io
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

import llm
import triage_images

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "bench" / "data"
IMAGES = DATA / "images"
ITEMS = DATA / "items.jsonl"
CROPS = DATA / "caption_crops.json"
RAW = REPO_ROOT / "bench" / "results" / "image_triage.jsonl"
BACKUP = REPO_ROOT / ".tmp" / "originals"

# A caption cannot take more than this share of the plate; past it the image is
# a report screenshot and cropping would delete the content instead of the text.
MAX_FRACTION = 0.6
MIN_FRACTION = 0.03
MIN_KEEP = 0.4

# The model's estimate is a starting point, not a measurement: on the pilot it
# came out 5 points short on a long legend, so the ladder keeps going deeper
# until the transcript stops naming the diagnosis.
LADDER = (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)

# A leftover caption line still names the diagnosis in its own words, which the
# character match may miss, so a long transcript also fails the check.
MAX_LEFTOVER_TEXT = 40

LOCATOR_PROMPT = """这张医学图片的上部或下部是否印有一段说明文字（期刊图注、检查报告文字、界面文字）？

只输出一个 JSON 对象：
{"caption": "bottom" | "top" | "none", "fraction": 0.0-0.6, "evidence": "不超过 20 字的依据"}

fraction 指这段成段文字大约占图片高度的比例（0 表示没有）。只算成段的文字，不算图上的箭头、标尺、部位字母、日期戳。
"""

_lock = threading.Lock()


def load_items():
    items = []

    with open(ITEMS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    return items


def load_records():
    """Read the triage output; a later line for the same image wins."""

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

            records[record["image"]] = record

    return records


def encode(image, png):
    """Encode a cropped image back to bytes, matching the original format."""

    buffer = io.BytesIO()

    if png:
        image.save(buffer, format="PNG", optimize=True)
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=92)

    return buffer.getvalue()


def crop(raw, fraction, side="bottom"):
    """Cut ``fraction`` of the height off one edge. Returns (image, bytes).

    Most plates print the legend under the figure, but a few put it on top, so
    the edge comes from the model and is part of the recorded decision.
    """

    image = Image.open(io.BytesIO(raw))
    width, height = image.size
    cut = int(round(height * fraction))
    cut = max(0, min(cut, height - 1))

    if side == "top":
        cropped = image.crop((0, cut, width, height))
    else:
        cropped = image.crop((0, 0, width, height - cut))

    return cropped, encode(cropped, raw.startswith(b"\x89PNG"))


def accepted(record, item, keep_ratio):
    """Decide whether the cropped image still leaks the answer."""

    text_kind = record.get("text_kind", "")
    text = (record.get("text") or "").strip()
    findings = (record.get("findings") or "").strip()
    leaks = triage_images.leaked_labels(record, item)

    if keep_ratio < MIN_KEEP:
        return False, "crop keeps too little of the plate"

    if findings in ("", "无法判读"):
        return False, "nothing readable left after the crop"

    if leaks:
        return False, f"gold label still printed on the image ({leaks[0][:20]})"

    if text_kind == "结论":
        return False, "the remaining text still states a conclusion"

    if len(text) > MAX_LEFTOVER_TEXT:
        return False, f"text still too long after the crop ({len(text)} chars)"

    return True, ""


def append_triage(relative, item_id, record):
    """Record the post-crop description so the triage report can be rebuilt."""

    entry = {
        "image": relative,
        "item_id": item_id,
        "modality": record.get("modality", ""),
        "findings": record.get("findings", ""),
        "text": record.get("text", ""),
        "text_kind": record.get("text_kind", ""),
        "error": None,
        "source": "cropped",
    }

    RAW.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process(item, relative, dry_run, raw_cache):
    """Locate the caption, crop, verify. Returns the record for the json."""

    path = IMAGES / relative

    with _lock:
        if relative not in raw_cache:
            raw_cache[relative] = path.read_bytes()
        raw = raw_cache[relative]

    image = Image.open(io.BytesIO(raw))
    width, height = image.size

    hint = triage_images.ask(LOCATOR_PROMPT, raw, max_tokens=300, label=relative)
    side = hint.get("caption")

    try:
        proposed = float(hint.get("fraction") or 0)
    except (TypeError, ValueError):
        proposed = 0.0

    proposed = max(0.0, min(proposed, MAX_FRACTION))

    if side not in ("bottom", "top") or proposed < MIN_FRACTION:
        return {
            "action": "skipped",
            "reason": (
                f"caption={side}, fraction={proposed:.2f}: the text is painted "
                "inside the image, cropping cannot remove it"
            ),
            "proposed_fraction": proposed,
        }

    if dry_run:
        return {
            "action": f"would-crop-{side}",
            "side": side,
            "proposed_fraction": proposed,
            "height_before": height,
            "height_after": height - int(round(height * proposed)),
            "size": [width, height],
        }

    attempts = []

    for step in LADDER:
        fraction = round(min(proposed + step, MAX_FRACTION), 3)

        if fraction < MIN_FRACTION:
            continue

        cropped, data = crop(raw, fraction, side)
        keep_ratio = cropped.size[1] / height

        try:
            record = triage_images.ask(triage_images.MODALITY_PROMPT, data, label=relative)
        except RuntimeError as error:
            attempts.append({"fraction": fraction, "ok": False, "why": str(error)[:120]})
            continue

        ok, why = accepted(record, item, keep_ratio)
        attempts.append({"fraction": fraction, "ok": ok, "why": why})

        if not ok:
            continue

        BACKUP.mkdir(parents=True, exist_ok=True)
        backup = BACKUP / relative
        backup.parent.mkdir(parents=True, exist_ok=True)

        if not backup.exists():
            shutil.copyfile(path, backup)

        staging = path.with_name(path.name + ".new")
        staging.write_bytes(data)
        os.replace(staging, path)

        append_triage(relative, item["item_id"], record)

        return {
            "action": "cropped",
            "side": side,
            "proposed_fraction": proposed,
            "applied_fraction": fraction,
            "height_before": height,
            "height_after": cropped.size[1],
            "attempts": attempts,
            "after": {
                "modality": record.get("modality", ""),
                "text": record.get("text", ""),
                "text_kind": record.get("text_kind", ""),
                "findings": record.get("findings", ""),
            },
        }

    return {
        "action": "skipped",
        "reason": "no crop removed the conclusion",
        "proposed_fraction": proposed,
        "attempts": attempts,
    }


def save(results):
    payload = {
        "note": (
            "Bottom/top caption crops applied in place to bench/data/images. "
            "Original bytes are kept in .tmp/originals/ and can be restored with "
            "bench/refresh_images.py."
        ),
        "model": triage_images.VISION_MODEL,
        "images": dict(sorted(results.items())),
    }

    CROPS.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="only ask where the captions are")
    args = parser.parse_args()

    if not llm.is_configured():
        raise SystemExit("no API key; put LLM_API_KEY in bench/.env")

    items = load_items()
    records = load_records()

    known = {}

    if CROPS.exists():
        with open(CROPS, encoding="utf-8") as f:
            known = json.load(f).get("images", {})

    targets = []
    seen = set()

    for item in items:
        for relative in item["images"]:

            if relative in seen:
                continue

            record = records.get(relative)

            if not record:
                continue

            if known.get(relative, {}).get("action") in ("cropped", "skipped"):
                continue

            leaks = triage_images.leaked_labels(record, item)

            if triage_images.tier(record, leaks) != "C":
                continue

            seen.add(relative)
            targets.append((item, relative))

    print(f"{len(targets)} image(s) to process", flush=True)

    if not targets:
        return

    results = dict(known)
    raw_cache = {}
    started = time.time()
    counter = {"n": 0}
    total = len(targets)

    def work(job):
        item, relative = job

        try:
            result = process(item, relative, args.dry_run, raw_cache)
        except Exception as exception:  # noqa: BLE001 - keep the batch alive
            result = {"action": "error", "reason": f"{type(exception).__name__}: {exception}"}

        with _lock:
            results[relative] = result
            counter["n"] += 1
            elapsed = time.time() - started

            if not args.dry_run:
                save(results)

            print(
                f"[{counter['n']}/{total}] {relative} {result.get('action')} "
                f"{result.get('applied_fraction') or result.get('proposed_fraction') or ''} "
                f"({elapsed / counter['n']:.0f}s each)",
                flush=True,
            )

        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, targets))

    if args.dry_run:
        return

    save(results)

    counts = {}

    for result in results.values():
        counts[result.get("action")] = counts.get(result.get("action"), 0) + 1

    print("actions:", json.dumps(counts, ensure_ascii=False))
    print("next: python bench/freeze_data.py --manifest-only")
    print("      python bench/triage_images.py --report-only")


if __name__ == "__main__":
    main()
